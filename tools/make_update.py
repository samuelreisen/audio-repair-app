"""Make the next update for the installed app: "Audio Repair - Mk <N> update.srar".

    python tools/make_update.py --title "Better declipping" --notes "What changed, for the person installing it"
                                [--entry entry.md] [--dry-run]

1. the next mark is the last one in updates/history.json + 1
2. HANDOFF.md gets this mark's entry at the top of its change log (section 12, the same Asked / Changed / Verified /
   Not done format as every other entry; the body comes from --entry <file>, otherwise from the title and notes), and
   its first line is stamped <!-- handoff-mark: N -->. An update therefore cannot be made without its handoff entry,
   and installing it brings every user's HANDOFF.md up to date.
3. mark.json is rewritten for the new mark (so installers built from now on have it built in)
4. the page (static/), engine.py, pipeline/*.py, mark.json and HANDOFF.md are packed into updates/out/<file>.srar
5. the file is checked with the same reader the app uses, and the mark is added to updates/history.json

Run it with any Python 3.9+ (only the standard library is used).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app_updates as up  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--title", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--entry", help="a file holding the full HANDOFF.md entry (Asked / Changed / Verified / Not done)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    hist_file = ROOT / "updates" / "history.json"
    hist = json.loads(hist_file.read_text(encoding="utf-8"))
    n = max(m["mark"] for m in hist["marks"]) + 1
    name, date, fname = up.mark_name(n), time.strftime("%Y-%m-%d"), up.file_name(n)

    # ---------------------------------------------------------------- HANDOFF.md entry
    handoff_path = ROOT / up.HANDOFF
    handoff = handoff_path.read_text(encoding="utf-8")
    if '– %s "' % name in handoff:
        sys.exit("HANDOFF.md already has an entry for %s. Remove it (or fix updates/history.json) first." % name)
    if a.entry:
        body = Path(a.entry).read_text(encoding="utf-8").strip()
    else:
        lines = [l.strip() for l in a.notes.splitlines() if l.strip()]
        changed = ("\n" + "\n".join("  " + (l if l.startswith("-") else "- " + l) for l in lines)) if lines else " see the notes."
        body = "\n".join(["- **Asked:** an update (%s): %s." % (name, a.title),
                          "- **Changed:**" + changed,
                          "- **Verified:** (not recorded: made without --entry; add how it was checked)",
                          "- **Not done:** (not recorded)"])
    body += ("\n- **Update file:** `%s`, needs app %s or newer. Installing it replaces the user's older HANDOFF.md "
             "with this one." % (fname, up.APP_VERSION))
    anchor = re.search(r"(?m)^Format: \*\*date – title\.\*\*.*$", handoff)
    if not anchor:
        sys.exit('HANDOFF.md: cannot find the "Format:" line of the change log.')
    heading = '### %s – %s "%s" – update file' % (date, name, a.title)
    handoff = handoff[:anchor.end()] + "\n\n" + heading + "\n" + body + handoff[anchor.end():]
    handoff = up.stamp_handoff(handoff, n)

    # ---------------------------------------------------------------- pack and check
    mark_json = json.dumps({"mark": n, "name": name, "title": a.title, "notes": a.notes, "created": date}) + "\n"
    files = up.project_files(ROOT)
    files["mark.json"] = mark_json.encode("utf-8")
    files[up.HANDOFF] = handoff.encode("utf-8")
    buf = up.pack({"mark": n, "title": a.title, "notes": a.notes, "created": date, "minApp": up.APP_VERSION}, files)
    h, back = up.parse(buf)
    assert back == files and h["mark"] == n
    print("%s – %s: %d files, %.2f MB, needs app %s+" % (name, a.title, len(files), len(buf) / 1048576, up.APP_VERSION))
    if a.dry_run:
        print("(dry run, nothing written; the HANDOFF.md entry would read:)\n" + heading + "\n" + body)
        return

    handoff_path.write_text(handoff, encoding="utf-8")
    (ROOT / "mark.json").write_text(mark_json, encoding="utf-8")
    out = ROOT / "updates" / "out" / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf)
    hist["marks"].append({"mark": n, "name": name, "title": a.title, "notes": a.notes, "created": date, "file": fname,
                          "sha256": hashlib.sha256(buf).hexdigest(), "bytes": len(buf), "minApp": up.APP_VERSION})
    hist_file.write_text(json.dumps(hist, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("HANDOFF.md updated (entry + handoff-mark %d)" % n)
    print(out)


if __name__ == "__main__":
    main()
