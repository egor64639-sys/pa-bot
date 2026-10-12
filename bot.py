"""
Pocket PA: a Telegram bot that turns scattered texts, voice notes and
screenshots into reminders, powered by Claude.

Run:  python bot.py   (see README.md for setup)
"""
import asyncio
import base64
import json
import logging
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes, MessageHandler, filters)

# ---------------------------------------------------------------- config
load_dotenv()
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID") or 0)
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
TZ = ZoneInfo(os.getenv("TIMEZONE", "Europe/London"))
BRIEF_TIME = os.getenv("MORNING_BRIEF", "08:00")
LECTURE_REMIND_MIN = int(os.getenv("LECTURE_REMIND_MINUTES", "15"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
DB_PATH = os.getenv("DB_PATH", "pa.db")

log = logging.getLogger("pa")
claude = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment


# ---------------------------------------------------------------- database
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            notes TEXT,
            kind TEXT DEFAULT 'todo',        -- 'todo' stays open until done; 'event' closes once it happens
            due_at TEXT,                     -- ISO datetime with offset; NULL = 'whenever'
            remind_before INTEGER DEFAULT 0, -- minutes
            recurrence TEXT DEFAULT 'none',  -- none | daily | weekdays | weekly
            repeat_until TEXT,
            status TEXT DEFAULT 'open',
            reminded INTEGER DEFAULT 0,
            snoozed_until TEXT,
            source TEXT DEFAULT 'chat',
            created_at TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT)")


def get_setting(key):
    with db() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else None


def set_setting(key, value):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, value))


# ---------------------------------------------------------------- time helpers
def now():
    return datetime.now(TZ)


def parse_local(s):
    """'2026-10-12T15:00' (local) or any ISO string -> aware datetime in TZ."""
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt.astimezone(TZ)


def fmt(iso):
    return parse_local(iso).strftime("%a %d %b, %H:%M") if iso else "no time"


def next_occurrence(due, rec, after):
    """Next repeat of `due` that falls after `after`, keeping local clock time across DST."""
    local = due.astimezone(TZ).replace(tzinfo=None)
    step = timedelta(days=7 if rec == "weekly" else 1)
    while True:
        local += step
        if rec == "weekdays" and local.weekday() >= 5:
            continue
        nxt = local.replace(tzinfo=TZ)
        if nxt > after:
            return nxt


# ---------------------------------------------------------------- task tools
def add_task(title, due=None, kind="todo", remind_before_minutes=0, recurrence="none",
             repeat_until=None, notes=None, source="chat"):
    due_at = parse_local(due).isoformat() if due else None
    with db() as c:
        cur = c.execute(
            "INSERT INTO tasks(title, notes, kind, due_at, remind_before, recurrence, repeat_until,"
            " source, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (title, notes, kind or "todo", due_at, remind_before_minutes or 0,
             recurrence or "none", repeat_until, source, now().isoformat()))
        return {"ok": True, "id": cur.lastrowid, "due": fmt(due_at) if due_at else None}


def update_task(id, **changes):
    cols = {"title": "title", "notes": "notes", "kind": "kind", "recurrence": "recurrence",
            "repeat_until": "repeat_until", "remind_before_minutes": "remind_before"}
    sets, vals = [], []
    for key, val in changes.items():
        if key == "due":
            sets += ["due_at=?", "reminded=0", "snoozed_until=NULL", "status='open'"]
            vals.append(parse_local(val).isoformat() if val else None)
        elif key in cols:
            sets.append(f"{cols[key]}=?")
            vals.append(val)
    if not sets:
        return {"ok": False, "error": "nothing to change"}
    with db() as c:
        n = c.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", (*vals, id)).rowcount
    return {"ok": bool(n)}


def complete_task(id):
    with db() as c:
        n = c.execute("UPDATE tasks SET status='done' WHERE id=?", (id,)).rowcount
    return {"ok": bool(n)}


def delete_task(id):
    with db() as c:
        n = c.execute("DELETE FROM tasks WHERE id=?", (id,)).rowcount
    return {"ok": bool(n)}


def task_line(r):
    when = fmt(r["due_at"]) if r["due_at"] else "whenever"
    extra = f", repeats {r['recurrence']}" if r["recurrence"] != "none" else ""
    note = f" ({r['notes']})" if r["notes"] else ""
    return f"#{r['id']} [{r['kind']}] {r['title']}{note}: {when}{extra}"


def list_tasks(status="open", days_ahead=None):
    with db() as c:
        rows = c.execute("SELECT * FROM tasks WHERE status=?", (status,)).fetchall()
    if days_ahead:
        cutoff = now() + timedelta(days=days_ahead)
        rows = [r for r in rows if not r["due_at"] or parse_local(r["due_at"]) <= cutoff]
    rows.sort(key=lambda r: parse_local(r["due_at"]) if r["due_at"] else datetime.max.replace(tzinfo=TZ))
    return [task_line(r) for r in rows[:150]]


TOOLS = [
    {"name": "add_task",
     "description": "Save a todo, event, or thing to remember. Call once per item.",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string", "description": "Short, clear title, e.g. 'Email Dr Patel about essay extension'"},
         "due": {"type": "string", "description": "Local datetime YYYY-MM-DDTHH:MM. Omit for 'whenever' items."},
         "kind": {"type": "string", "enum": ["todo", "event"],
                  "description": "todo = something to do (stays open until done). event = something happening at a time (lecture, appointment)."},
         "remind_before_minutes": {"type": "integer", "description": "Remind this many minutes before due. 0 = at due time."},
         "recurrence": {"type": "string", "enum": ["none", "daily", "weekdays", "weekly"]},
         "repeat_until": {"type": "string", "description": "YYYY-MM-DD, last date for a repeating item"},
         "notes": {"type": "string", "description": "Extra detail, e.g. location or who it's for"}},
         "required": ["title"]}},
    {"name": "update_task",
     "description": "Change an existing task (reschedule, rename, etc.). Only pass fields that change.",
     "input_schema": {"type": "object", "properties": {
         "id": {"type": "integer"}, "title": {"type": "string"}, "due": {"type": "string"},
         "kind": {"type": "string", "enum": ["todo", "event"]},
         "remind_before_minutes": {"type": "integer"},
         "recurrence": {"type": "string", "enum": ["none", "daily", "weekdays", "weekly"]},
         "repeat_until": {"type": "string"}, "notes": {"type": "string"}},
         "required": ["id"]}},
    {"name": "complete_task", "description": "Mark a task as done.",
     "input_schema": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}},
    {"name": "delete_task", "description": "Delete a task that was cancelled or added by mistake.",
     "input_schema": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}},
    {"name": "list_tasks",
     "description": "List tasks beyond the snapshot in your instructions, e.g. completed ones or further ahead.",
     "input_schema": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["open", "done"]},
         "days_ahead": {"type": "integer", "description": "Only items due within this many days (plus undated)"}}}},
]

TOOL_FUNCS = {"add_task": add_task, "update_task": update_task, "complete_task": complete_task,
              "delete_task": delete_task, "list_tasks": list_tasks}


# ---------------------------------------------------------------- Claude
SYSTEM = """You are a warm, concise personal assistant (PA) for a forgetful, busy Cambridge student.
They send you scattered texts, rambling voice notes and screenshots throughout the day. Your job:

- Pull out EVERY actionable item, event or thing-to-remember, even if buried mid-ramble, and save each one
  with add_task. One message often contains several items. Ignore filler and thinking-out-loud.
- Turn relative times ("tomorrow arvo", "after my 11 o'clock", "next Weds") into exact local datetimes
  using the current time below. Defaults: morning 09:00, afternoon 14:00, evening 18:00, tonight 20:00.
- If no time is given and it isn't urgent, save it with no due time (it goes on their 'whenever' list,
  shown in the morning brief). If a time is needed to be useful, pick a sensible one and say so.
- Events (lectures, appointments, meetings): kind=event, remind ~15 min before, or 30-60 if they need to travel.
  Deadlines and chores: kind=todo, remind at a useful moment before the deadline, not just at it.
- Check the task list first; update instead of creating duplicates.
- When they say something is done, mark it complete. Match loosely ("sent the email" = the email task).
- Voice transcripts can contain mistakes; use common sense about what they meant.
- Only ask a question if you truly can't act. Otherwise act, state your assumption, and let them correct you.
- Reply in plain text (no markdown, no asterisks), short and phone-sized. Confirm what you saved, with times.

Current time: {now}

Open tasks (overdue, next 14 days, and undated):
{tasks}"""

HISTORY = []        # recent text-only turns, so follow-ups like "make that 4pm" work
MAX_HISTORY = 20


def build_system():
    tasks = list_tasks("open", days_ahead=14)
    return SYSTEM.format(now=now().strftime("%A %d %B %Y, %H:%M"),
                         tasks="\n".join(tasks) if tasks else "(none yet)")


def run_tool(name, args):
    try:
        return TOOL_FUNCS[name](**args)
    except Exception as e:  # report errors back to Claude so it can fix its call
        return {"ok": False, "error": str(e)}


async def ask_claude(content, history_text):
    messages = HISTORY[-MAX_HISTORY:] + [{"role": "user", "content": content}]
    for _ in range(10):
        resp = await claude.messages.create(model=MODEL, max_tokens=1500, system=build_system(),
                                            tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = [{"type": "tool_result", "tool_use_id": b.id,
                    "content": json.dumps(run_tool(b.name, b.input))}
                   for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "user", "content": results})
    reply = "".join(b.text for b in resp.content if b.type == "text").strip() or "Done ✅"
    HISTORY.extend([{"role": "user", "content": history_text},
                    {"role": "assistant", "content": reply}])
    del HISTORY[:-MAX_HISTORY]
    return reply


# ---------------------------------------------------------------- Telegram handlers
def allowed(update: Update):
    return ALLOWED_USER_ID and update.effective_user and update.effective_user.id == ALLOWED_USER_ID


async def gatekeep(update: Update):
    """Returns True if this user may use the bot. Helps with first-time setup."""
    if allowed(update):
        return True
    if not ALLOWED_USER_ID and update.message:
        await update.message.reply_text(
            f"Hi! Your Telegram user ID is {update.effective_user.id}.\n"
            f"Put ALLOWED_USER_ID={update.effective_user.id} in your .env file and restart me.")
    return False


async def respond(update: Update, context, content, history_text):
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    try:
        reply = await ask_claude(content, history_text)
    except Exception:
        log.exception("Claude call failed")
        reply = "Sorry, something went wrong on my end. Try that again in a moment?"
    await update.message.reply_text(reply)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await gatekeep(update):
        await respond(update, context, update.message.text, update.message.text)


_whisper = None


def transcribe(path):
    global _whisper
    if _whisper is None:  # loaded on first use; downloads the model once
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(path)
    return " ".join(s.text.strip() for s in segments).strip()


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await gatekeep(update):
        return
    audio = update.message.voice or update.message.audio
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    tg_file = await audio.get_file()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "note.ogg")
        await tg_file.download_to_drive(path)
        text = await asyncio.to_thread(transcribe, path)
    if not text:
        await update.message.reply_text("I couldn't make that out. Could you try again?")
        return
    await update.message.reply_text(f"🎙️ Heard: {text}")
    await respond(update, context, f"[Voice note, auto-transcribed] {text}", f"[voice] {text}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await gatekeep(update):
        return
    tg_file = await update.message.photo[-1].get_file()
    data = base64.standard_b64encode(bytes(await tg_file.download_as_bytearray())).decode()
    caption = update.message.caption or (
        "Here's a screenshot. Pull out anything I need to remember (timetable entries, "
        "deadlines, events) and add them.")
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
               {"type": "text", "text": caption}]
    await respond(update, context, content, f"[sent a photo] {caption}")


# ---------------------------------------------------------------- reminders
def reminder_time(r):
    if r["snoozed_until"]:
        return parse_local(r["snoozed_until"])
    return parse_local(r["due_at"]) - timedelta(minutes=r["remind_before"] or 0)


def reminder_buttons(task_id):
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Done", callback_data=f"done:{task_id}"),
        InlineKeyboardButton("⏰ 15 min", callback_data=f"snooze:{task_id}:15"),
        InlineKeyboardButton("⏰ 1 hour", callback_data=f"snooze:{task_id}:60"),
    ]])


async def check_reminders(context: ContextTypes.DEFAULT_TYPE):
    t = now()
    with db() as c:
        rows = c.execute("SELECT * FROM tasks WHERE status='open' AND due_at IS NOT NULL"
                         " AND reminded=0").fetchall()
    for r in rows:
        if reminder_time(r) > t:
            continue
        due = parse_local(r["due_at"])
        mins = round((due - t).total_seconds() / 60)
        when = (f"in {mins} min" if mins > 0 else "now" if mins > -10
                else f"was due {fmt(r['due_at'])}")
        text = f"🔔 {r['title']}, {when}" + (f"\n📝 {r['notes']}" if r["notes"] else "")
        await context.bot.send_message(ALLOWED_USER_ID, text, reply_markup=reminder_buttons(r["id"]))

        with db() as c:
            if r["recurrence"] != "none":
                nxt = next_occurrence(due, r["recurrence"], after=t)
                until = r["repeat_until"]
                if until and nxt.date() > date.fromisoformat(until[:10]):
                    c.execute("UPDATE tasks SET status='done', reminded=1 WHERE id=?", (r["id"],))
                else:
                    c.execute("UPDATE tasks SET due_at=?, reminded=0, snoozed_until=NULL WHERE id=?",
                              (nxt.isoformat(), r["id"]))
            else:
                # events close themselves once reminded; todos stay open until you tap Done
                status = "done" if r["kind"] == "event" else "open"
                c.execute("UPDATE tasks SET reminded=1, snoozed_until=NULL, status=? WHERE id=?",
                          (status, r["id"]))


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if q.from_user.id != ALLOWED_USER_ID:
        await q.answer()
        return
    parts = q.data.split(":")
    action, tid = parts[0], int(parts[1])
    with db() as c:
        r = c.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
    await q.answer()
    if not r:
        await q.edit_message_text("That one's already been removed.")
        return
    recurring = r["recurrence"] != "none"
    if action == "done":
        if not recurring:
            complete_task(tid)
        await q.edit_message_text(f"✅ {r['title']}")
    elif action == "snooze":
        until = now() + timedelta(minutes=int(parts[2]))
        if recurring:  # snooze just this occurrence by making a one-off copy
            add_task(r["title"], due=until.isoformat(), notes=r["notes"])
        else:
            with db() as c:
                c.execute("UPDATE tasks SET snoozed_until=?, reminded=0, status='open' WHERE id=?",
                          (until.isoformat(), tid))
        await q.edit_message_text(f"⏰ {r['title']}, snoozed until {until:%H:%M}")


# ---------------------------------------------------------------- morning brief
def build_brief():
    t = now()
    with db() as c:
        rows = c.execute("SELECT * FROM tasks WHERE status='open'").fetchall()
    today, overdue, whenever = [], [], []
    for r in rows:
        if not r["due_at"]:
            whenever.append(r)
            continue
        d = parse_local(r["due_at"])
        if d.date() == t.date():
            today.append((d, r))
        elif d < t and r["kind"] == "todo":
            overdue.append((d, r))
    lines = [f"☀️ Good morning! Here's {t:%A %d %B}."]
    if today:
        lines.append("\nToday:")
        lines += [f"• {d:%H:%M} {r['title']}" + (f" ({r['notes']})" if r["notes"] else "")
                  for d, r in sorted(today, key=lambda x: x[0])]
    if overdue:
        lines.append("\nStill to do (overdue):")
        lines += [f"• {r['title']} (was {d:%a %d %b})" for d, r in sorted(overdue, key=lambda x: x[0])]
    if whenever:
        lines.append("\nWhenever you get a chance:")
        lines += [f"• {r['title']}" for r in whenever]
    if len(lines) == 1:
        lines.append("Nothing on the list today. Send me anything that pops into your head.")
    return "\n".join(lines)


async def morning_brief(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(ALLOWED_USER_ID, build_brief())


# ---------------------------------------------------------------- Cambridge timetable (iCal)
def import_ics(ics_bytes, days=120):
    """Replace upcoming, not-yet-reminded timetable events with what's in the feed."""
    import icalendar
    import recurring_ical_events
    cal = icalendar.Calendar.from_ical(ics_bytes)
    t = now()
    events = recurring_ical_events.of(cal).between(t, t + timedelta(days=days))
    with db() as c:
        c.execute("DELETE FROM tasks WHERE source='timetable' AND status='open' AND reminded=0")
    count = 0
    for ev in events:
        start = ev.get("DTSTART").dt
        if not isinstance(start, datetime):  # skip all-day entries
            continue
        start = start.replace(tzinfo=TZ) if start.tzinfo is None else start.astimezone(TZ)
        if start - timedelta(minutes=LECTURE_REMIND_MIN) <= t:
            continue
        add_task(str(ev.get("SUMMARY", "Lecture")), due=start.isoformat(), kind="event",
                 remind_before_minutes=LECTURE_REMIND_MIN,
                 notes=str(ev.get("LOCATION") or "") or None, source="timetable")
        count += 1
    return count


async def sync_timetable(url):
    url = url.strip().replace("webcal://", "https://", 1)
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as h:
        resp = await h.get(url)
        resp.raise_for_status()
    return import_ics(resp.content)


async def sync_timetable_job(context: ContextTypes.DEFAULT_TYPE):
    url = get_setting("timetable_url") or os.getenv("TIMETABLE_ICS_URL")
    if url:
        try:
            n = await sync_timetable(url)
            log.info("Timetable synced: %d events", n)
        except Exception:
            log.exception("Timetable sync failed")


# ---------------------------------------------------------------- commands
HELP = ("I'm your PA. Just message me, or send a voice note, whenever something pops into your head:\n"
        "• \"dentist thursday 3pm\"\n"
        "• \"oh and I need to return the library book and email my supervisor\"\n"
        "• \"done the email\" / \"move the dentist to 4\"\n"
        "• a screenshot of anything with dates on it\n\n"
        "/today: today's plan\n"
        "/timetable <link>: import your Cambridge lecture timetable")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await gatekeep(update):
        await update.message.reply_text(HELP)


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await gatekeep(update):
        await update.message.reply_text(build_brief())


async def cmd_timetable(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await gatekeep(update):
        return
    url = context.args[0] if context.args else get_setting("timetable_url") or os.getenv("TIMETABLE_ICS_URL")
    if not url:
        await update.message.reply_text(
            "Send /timetable followed by your calendar link from timetable.cam.ac.uk "
            "(the 'connect to calendar' option).")
        return
    try:
        n = await sync_timetable(url)
    except Exception as e:
        await update.message.reply_text(f"Couldn't import that link: {e}")
        return
    set_setting("timetable_url", url)
    await update.message.reply_text(
        f"📚 Imported {n} upcoming lectures. I'll remind you {LECTURE_REMIND_MIN} min before each, "
        "and re-check the timetable every morning for changes.")


# ---------------------------------------------------------------- main
async def on_error(update, context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error", exc_info=context.error)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not TELEGRAM_TOKEN:
        raise SystemExit("TELEGRAM_TOKEN is missing. Copy .env.example to .env and fill it in.")
    init_db()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("timetable", cmd_timetable))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(on_error)

    if ALLOWED_USER_ID:
        hh, mm = map(int, BRIEF_TIME.split(":"))
        app.job_queue.run_repeating(check_reminders, interval=30, first=5)
        app.job_queue.run_daily(morning_brief, time=dtime(hh, mm, tzinfo=TZ))
        app.job_queue.run_daily(sync_timetable_job, time=dtime(6, 0, tzinfo=TZ))
    else:
        log.warning("ALLOWED_USER_ID not set: message the bot to get your ID.")

    log.info("PA bot running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
