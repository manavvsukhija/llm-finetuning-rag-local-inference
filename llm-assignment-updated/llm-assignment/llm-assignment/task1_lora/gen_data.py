"""Generate a synthetic 'plain English -> fixed JSON' dataset.
Train uses only the 'seen' templates. Test mixes seen templates with HELD-OUT templates,
unseen names and unseen tasks, so the fine-tuned model has realistic ways to fail.
Usage: python gen_data.py   (writes train.jsonl, test.jsonl)"""
import json, random

NAMES = ["Aarav", "Priya", "Rohan", "Meera", "Karan", "Sneha", "Vikram", "Ananya", "Dev", "Isha"]
EXTRA_NAMES = ["Tanvi", "Harsh", "Naman"]          # test only
TASKS = ["call the plumber", "submit the assignment", "buy groceries", "pay the electricity bill",
         "water the plants", "book train tickets", "renew my passport", "email the report",
         "go to the gym", "prepare for the viva"]
EXTRA_TASKS = ["pick up the parcel", "back up my laptop"]   # test only
WHENS = ["tomorrow at 9 am", "on Friday evening", "next Monday at 10:30", "tonight at 8",
         "in two hours", "on 25 December", "this Sunday morning", "at 6 pm today",
         "next week", "on the 3rd at noon"]
ITEMS = [("coffee", "food"), ("auto ride", "transport"), ("movie ticket", "entertainment"),
         ("notebook", "stationery"), ("pizza", "food"), ("metro card recharge", "transport"),
         ("headphones", "shopping"), ("doctor visit", "health"), ("phone recharge", "bills"),
         ("textbook", "education")]
MSGS = ["I will be late", "the meeting is moved to 4", "happy birthday", "please send the notes",
        "call me when free", "I reached home", "lunch at 1?", "the files are uploaded"]

T = {
 "set_reminder": {
  "seen": ["Remind me to {task} {when}", "Set a reminder: {task} {when}",
           "Can you remind me to {task} {when}?", "{when}, remind me to {task}"],
  "heldout": ["Don't let me forget to {task} {when}", "I need a nudge to {task} {when}, please"]},
 "add_expense": {
  "seen": ["I spent {amt} rupees on {item}", "Add an expense of {amt} for {item}", "Log {amt} rs {item}"],
  "heldout": ["Paid ₹{amt} for {item} just now", "{item} cost me {amt} bucks"]},
 "send_message": {
  "seen": ["Text {name} that {msg}", "Send a message to {name}: {msg}", "Tell {name} {msg}"],
  "heldout": ["Drop {name} a note saying {msg}", "WhatsApp {name} - {msg}"]},
}

def sample(rng, heldout, test):
    intent = rng.choice(list(T))
    t = rng.choice(T[intent]["heldout" if heldout else "seen"])
    if intent == "set_reminder":
        task = rng.choice(TASKS + (EXTRA_TASKS if test else [])); when = rng.choice(WHENS)
        text = t.format(task=task, when=when)
        obj = {"intent": intent, "task": task, "when": when}
    elif intent == "add_expense":
        item, cat = rng.choice(ITEMS)
        amt = rng.randint(20, 5000) if rng.random() < .6 else round(rng.uniform(10, 999), 1)
        text = t.format(amt=amt, item=item)
        obj = {"intent": intent, "amount": amt, "category": cat, "item": item}
    else:
        name = rng.choice(NAMES + (EXTRA_NAMES if test else [])); msg = rng.choice(MSGS)
        text = t.format(name=name, msg=msg)
        obj = {"intent": intent, "recipient": name, "message": msg}
    if rng.random() < .15: text = text.lower()
    return {"text": text, "target": json.dumps(obj, ensure_ascii=False), "heldout_template": heldout}

def build(n, heldout_frac, test, rng, exclude=()):
    seen, out = set(exclude), []
    while len(out) < n:
        ex = sample(rng, rng.random() < heldout_frac, test)
        if ex["text"] in seen: continue
        seen.add(ex["text"]); out.append(ex)
    return out

if __name__ == "__main__":
    rng = random.Random(7)
    train = build(1200, 0.0, False, rng)
    test = build(100, 0.5, True, rng, exclude=[e["text"] for e in train])
    for name, rows in (("train", train), ("test", test)):
        with open(f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(train), "train /", len(test), "test examples written")
