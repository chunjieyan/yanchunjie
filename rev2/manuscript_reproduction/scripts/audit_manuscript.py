"""Strict structural, bibliographic, asset, and numeric integrity audit for manuscript_rev2."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_all_tex():
    main = (ROOT / "main.tex").read_text(encoding="utf-8")
    # Expand \input{...} recursively (non-commented)
    inputs = []

    def expand(text, depth=0):
        out = []
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("%"):
                continue
            # Strip trailing inline comments (naive but safe for non-escaped %)
            code = re.sub(r"(?<!\\)%.*$", "", line)
            m = re.search(r"\\input\{([^}]+)\}", code)
            if m:
                rel = m.group(1)
                if not rel.endswith(".tex"):
                    rel += ".tex"
                p = ROOT / rel
                inputs.append((rel, p.is_file()))
                if p.is_file() and depth < 5:
                    out.append(expand(p.read_text(encoding="utf-8"), depth + 1))
            else:
                out.append(code)
        return "\n".join(out)

    full_tex = expand(main)
    return full_tex, inputs


def main():
    full_tex, inputs = read_all_tex()
    errors = []

    # 1. Check all \input files exist
    missing_inputs = [rel for rel, ok in inputs if not ok]
    print(f"[INPUTS] Checked {len(inputs)} \\input files; missing: {len(missing_inputs)} {missing_inputs}")
    if missing_inputs:
        errors.append(f"Missing inputs: {missing_inputs}")

    # 2. Check all \includegraphics files exist
    gfx = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", full_tex)
    missing_gfx = [g for g in gfx if not (ROOT / g).is_file()]
    print(f"[FIGURES] Checked {len(gfx)} \\includegraphics files; missing: {len(missing_gfx)} {missing_gfx}")
    if missing_gfx:
        errors.append(f"Missing figures: {missing_gfx}")

    # 3. Check all \cite keys vs references.bib
    bib_txt = (ROOT / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib_txt))
    cited_keys = set()
    for m in re.findall(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]+)\}", full_tex):
        for k in m.split(","):
            k = k.strip()
            if k:
                cited_keys.add(k)
    undef_cites = sorted(cited_keys - bib_keys)
    uncited_bibs = sorted(bib_keys - cited_keys)
    print(f"[BIB] BibTeX entries: {len(bib_keys)} | Cited keys: {len(cited_keys)} | Undefined: {len(undef_cites)} | Uncited: {len(uncited_bibs)}")
    if undef_cites:
        print("  UNDEFINED CITES:", undef_cites)
        errors.append(f"Undefined cites: {undef_cites}")
    if uncited_bibs:
        print("  UNCITED BIB KEYS:", uncited_bibs)
        errors.append(f"Uncited bib keys: {uncited_bibs}")

    # 4. Check all \ref / \eqref vs \label
    labels = set(re.findall(r"\\label\{([^}]+)\}", full_tex))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", full_tex))
    undef_refs = sorted(refs - labels)
    print(f"[REFS] Labels: {len(labels)} | Refs: {len(refs)} | Undefined refs: {len(undef_refs)} {undef_refs}")
    if undef_refs:
        errors.append(f"Undefined refs: {undef_refs}")

    # 5. Check balanced \begin{env} / \end{env}
    begins = re.findall(r"\\begin\{([^}]+)\}", full_tex)
    ends = re.findall(r"\\end\{([^}]+)\}", full_tex)
    from collections import Counter
    cb, ce = Counter(begins), Counter(ends)
    env_mismatch = {k: (cb[k], ce[k]) for k in set(cb) | set(ce) if cb[k] != ce[k]}
    print(f"[ENVS] Balanced check: mismatches = {env_mismatch}")
    if env_mismatch:
        errors.append(f"Environment mismatches: {env_mismatch}")

    # 6. Check figure/table numbering continuity
    main_only = (ROOT / "main.tex").read_text(encoding="utf-8")
    body, app = main_only.split(r"\begin{appendices}")
    # Count figures in body
    body_figs = len(re.findall(r"\\begin\{figure", body))
    m_fig_ctr = re.search(r"\\setcounter\{figure\}\{(\d+)\}", app)
    fig_ctr = int(m_fig_ctr.group(1)) if m_fig_ctr else -1
    # Count tables in body (table inputs)
    body_tab_inputs = re.findall(r"\\input\{(tables/[^}]+)\}", body)
    m_tab_ctr = re.search(r"\\setcounter\{table\}\{(\d+)\}", app)
    tab_ctr = int(m_tab_ctr.group(1)) if m_tab_ctr else -1
    print(f"[COUNTERS] Main figs={body_figs} (app counter={fig_ctr}) | Main table inputs={len(body_tab_inputs)} (app counter={tab_ctr})")
    if body_figs != fig_ctr:
        errors.append(f"Figure counter mismatch: body={body_figs} vs app setcounter={fig_ctr}")
    if len(body_tab_inputs) != tab_ctr:
        errors.append(f"Table counter mismatch: body={len(body_tab_inputs)} vs app setcounter={tab_ctr}")

    # 7. Check no \nocite
    if r"\nocite" in full_tex:
        errors.append("Found forbidden \\nocite!")

    # 8. Check no raw unescaped '#' in prose/captions
    bad_hash = []
    for idx_l, line in enumerate(main_only.splitlines(), 1):
        s = line.lstrip()
        if s.startswith("%") or "newcommand" in s or "patchcmd" in s:
            continue
        code = re.sub(r"(?<!\\)%.*$", "", line)
        if re.search(r"(?<!\\)#\d", code):
            bad_hash.append((idx_l, code.strip()))
    if bad_hash:
        print(f"[WARN/ERR] Unescaped # in main.tex: {bad_hash}")
        errors.append(f"Unescaped # in main.tex: {bad_hash}")

    if errors:
        print("\nAUDIT FAILED:", errors)
        raise SystemExit(1)
    print("\nALL MANUSCRIPT INTEGRITY AUDITS PASSED (0 ERRORS)!")


if __name__ == "__main__":
    main()
