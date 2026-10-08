"""文档守卫：链接可达、权威声明、参数定值与代码一致、索引完整。

    python tools/audit_docs.py

**为什么需要它**：本仓库踩过的坑几乎都是「文档里的数字悄悄过期」——
`0.62` 被当成现行采纳值、`50/31` 上界、`0.0224` 权重余量、附录 A.3 的超窗反例。
散文错误容易看见，**数字漂移看不见**。本工具把四件事变成可执行断言：

1. **链接可达**：`docs/**/*.md` 与根 `README.md` 里所有相对链接都必须指向存在的路径
   （归档移动、改名会立刻暴露）。
2. **赋值式一致**：形如 `TIME_FLOOR = 0.605` / `THETA = 0.3943` 的写法，
   数字必须等于 `moonshadow.scoring` 里的常量。当前文档出现赋值即受检。
3. **参数表一致**：`v1.2-spec.md` 参数表里以这些标识符开头的行必须含当前值。
4. **权威声明与索引完整**：每份现行文档都要声明与 `v1.2-spec.md` 的关系；
   每份 `.md`（含归档）都必须出现在 `docs/README.md` 的清单里——
   新增文档忘了登记索引会被判违规。

**归档目录豁免数值检查**：`docs/archive/` 与 `docs/v1.0-moonshadow.md` 是**冻结的历史**，
它们写的是当时的取值（例如 `f = 0.62`），今天看「不一致」是正常的，
改它们才违反「归档即冻结」。归档只受链接可达约束。

退出码：无违规为 `0`，否则为 `1`。
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.scoring import ADOPTED_SCHEME, THETA, THETA_HIGH, TIME_FLOOR  # noqa: E402

DOCS = ROOT / "docs"
INDEX = DOCS / "README.md"
ROOT_README = ROOT / "README.md"

#: 唯一权威文档。
SPEC = DOCS / "v1.2-spec.md"

#: **现行版本前缀**。升版时要把上一个前缀挪进 `HISTORICAL_PREFIXES`——有意动作，
#: 不能靠「新文件自动算现行」蒙过去。
CURRENT_PREFIX = "v1.2"

#: 历史版本前缀。这些文档**保留**（记录推理过程：当时为什么那样想、哪条路被否决、
#: 错误怎么被发现的），但其中数据**不作为依据**，故豁免数值检查并强制状态横幅。
HISTORICAL_PREFIXES = ("v1.0", "v1.1")

#: **不随版本走的活文档**（索引、想法池）。它们不是某个版本的快照，而是持续维护的入口，
#: 因此按「现行」对待：必须声明与 spec 的关系，且受数值检查约束。
#: 新增此类文档必须显式登记在这里——否则会被 `check_classification` 拦下。
LIVING = ("README.md", "backlog.md")

#: 历史文档必须带的状态标注。
HISTORICAL_MARKERS = ("历史版本", "不作为当前依据")

#: 现行文档引用历史文档时，同一行必须出现的语义标记——
#: 防止把历史数字当依据读（只写版本号不算，那太容易误过）。
CITATION_MARKERS = ("历史", "思考过程", "已归档", "取代", "当时", "失效", "被否", "旧")

#: 现行文档（除 spec 本身）：每份都必须声明与 spec 的关系。
CURRENT = (
    DOCS / "v1.2-weights.md",
    DOCS / "v1.2-summary.md",
    DOCS / "v1.2-candidates.md",
    DOCS / "v1.2-roadmap.md",
    INDEX,
    DOCS / "backlog.md",
    ROOT_README,
)

#: 赋值式：`IDENT = <number>`。只认这一种形式——「提及标识符」不算声称取值，
#: 否则 `THETA_HIGH == THETA`、符号表、`f = 0.62（历史列）` 都会误报。
_ASSIGN = re.compile(r"\b(TIME_FLOOR|THETA_HIGH|THETA)\s*=\s*([0-9]+(?:\.[0-9]+)?)")
#: 参数表行：以 ``| `IDENT` `` 开头的表格单元格。
_TABLE_ROW = re.compile(r"^\|\s*`?(TIME_FLOOR|THETA_HIGH|THETA|ADOPTED_SCHEME)")
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
#: 历史文档的文件名（正文里提到即算一次引用）。
_HISTORICAL_NAME = re.compile(r"v1\.[01]-[a-z0-9\-]+\.md")
#: 反引号里以 `.md` 结尾的路径文本（不是 markdown 链接，因此链接检查抓不到）。
_INLINE_DOC_PATH = re.compile(r"`([^`\s]+\.md)`")
#: 反引号里指向 `docs/` 或 `archive/` 的路径文本——目录改名/删除同样会留下死引用。
_INLINE_DIR_PATH = re.compile(r"`((?:docs|archive)/[^`\s]*)`")

VALUES = {"TIME_FLOOR": TIME_FLOOR, "THETA_HIGH": THETA_HIGH, "THETA": THETA}


def markdown_files() -> list[pathlib.Path]:
    return sorted(
        list(DOCS.rglob("*.md")) + ([ROOT_README] if ROOT_README.exists() else [])
    )


def lines_of(path: pathlib.Path) -> list[str]:
    """按 `\\n` 切分。

    不用 `str.splitlines()`：它还会在 `U+2028`、垂直制表符等字符处断行，
    行号会与编辑器和 `read` 工具不一致，报错信息里的行号就指不准了。
    """
    return path.read_text(encoding="utf-8").split("\n")


def is_historical(path: pathlib.Path) -> bool:
    return path.name.startswith(HISTORICAL_PREFIXES)


def is_living(path: pathlib.Path) -> bool:
    """不随版本走的活文档（索引、想法池）。"""
    return path.parent == DOCS and path.name in LIVING


def check_links() -> list[str]:
    violations: list[str] = []
    for path in markdown_files():
        for target in _LINK.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            pure = target.split("#")[0]
            if not pure:
                continue
            if not (path.parent / pure).resolve().exists():
                violations.append(f"{path.relative_to(ROOT)}：链接失效 → {target}")
    return violations


def check_values() -> list[str]:
    """现行文档里不得出现与代码不符的赋值式。"""
    violations: list[str] = []
    for path in markdown_files():
        if is_historical(path):
            continue
        for number, line in enumerate(lines_of(path), 1):
            for identifier, literal in _ASSIGN.findall(line):
                if float(literal) != VALUES[identifier]:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{number}：{identifier} = {literal}"
                        f"，而代码里是 {VALUES[identifier]}"
                    )
    return violations


def check_classification() -> list[str]:
    """`docs/` 下每份文档都必须被显式分类：索引 / 活文档 / 现行版本 / 历史版本。

    新增文档时若既不是现行前缀、也不是已登记的历史前缀或活文档，就会在这里被拦下——
    迫使「升版」与「新增活文档」成为有意动作（更新 `HISTORICAL_PREFIXES` / `LIVING`）。
    """
    violations: list[str] = []
    for path in sorted(DOCS.rglob("*.md")):
        if path.name.startswith(CURRENT_PREFIX) or is_historical(path) or is_living(path):
            continue
        violations.append(
            f"{path.relative_to(ROOT)}：未分类。现行前缀是 {CURRENT_PREFIX}，"
            f"历史前缀是 {HISTORICAL_PREFIXES}，活文档是 {LIVING}；"
            f"新版本或新活文档请更新 audit_docs.py 的分类表"
        )
    return violations


def check_historical_banners() -> list[str]:
    """历史版本必须带状态横幅：说明它是「思考过程记录」且数据不作为依据。"""
    violations: list[str] = []
    for path in markdown_files():
        if not is_historical(path):
            continue
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in HISTORICAL_MARKERS if marker not in text]
        if missing:
            violations.append(
                f"{path.relative_to(ROOT)}：历史文档缺状态标注 {missing}"
                f"（保留它可以，但必须写明不作为依据）"
            )
    return violations


def check_citations_of_history() -> list[str]:
    """现行文档提到历史文档时，**附近**必须有语义标记。

    用 ±2 行的窗口而不是同一行：引用常常跨行（多行引用块、ASCII 图、表格换行），
    要求同行会制造大量假违规。窗口内出现「历史 / 思考过程 / 已归档 / 取代 / 当时 / 失效」
    即可——只写版本号不算标记，版本号太容易顺带出现。
    """
    violations: list[str] = []
    window = 2
    for path in markdown_files():
        if is_historical(path):
            continue
        lines = lines_of(path)
        for index, line in enumerate(lines):
            if not _HISTORICAL_NAME.search(line):
                continue
            nearby = "\n".join(lines[max(0, index - window) : index + window + 1])
            if any(marker in nearby for marker in CITATION_MARKERS):
                continue
            violations.append(
                f"{path.relative_to(ROOT)}:{index + 1}：引用了历史文档但附近未标注其历史地位"
                f" → {line.strip()[:80]}"
            )
    return violations


def check_parameter_table() -> list[str]:
    """`v1.2-spec.md` 参数表必须以标识符开头的行含当前值。"""
    violations: list[str] = []
    if not SPEC.exists():
        return [f"缺少权威文档 {SPEC.relative_to(ROOT)}"]
    for number, line in enumerate(lines_of(SPEC), 1):
        match = _TABLE_ROW.match(line)
        if not match:
            continue
        identifier = match.group(1)
        expected = ADOPTED_SCHEME if identifier == "ADOPTED_SCHEME" else VALUES[identifier]
        if str(expected) not in line:
            violations.append(
                f"{SPEC.relative_to(ROOT)}:{number}：参数表行 {identifier} 未含当前值 {expected}"
                f" → {line.strip()[:90]}"
            )
    return violations


def check_authority() -> list[str]:
    violations: list[str] = []
    if not SPEC.exists():
        return [f"缺少权威文档 {SPEC.relative_to(ROOT)}"]
    if "本文件是契约" not in SPEC.read_text(encoding="utf-8"):
        violations.append(f"{SPEC.relative_to(ROOT)}：未声明自己是契约（缺「本文件是契约」）")
    for path in CURRENT:
        if not path.exists():
            violations.append(f"声明的现行文档不存在：{path.relative_to(ROOT)}")
            continue
        if SPEC.name not in path.read_text(encoding="utf-8"):
            violations.append(
                f"{path.relative_to(ROOT)}：未声明与 {SPEC.name} 的关系"
                f"（现行文档必须指向唯一权威）"
            )
    return violations


def check_inline_doc_paths() -> list[str]:
    """反引号里的文档路径必须存在。

    这是实测漏过的一类错：正文写「已归档到 `docs/archive/`」，而目录已不存在——
    它不是 markdown 链接，所以链接检查看不见。路径漂移必须同样被拦住。
    """
    violations: list[str] = []
    for path in markdown_files():
        for number, line in enumerate(lines_of(path), 1):
            for token in _INLINE_DOC_PATH.findall(line) + _INLINE_DIR_PATH.findall(line):
                if "*" in token or "{" in token:
                    continue
                candidates = [ROOT / token, DOCS / token, path.parent / token]
                if not any(candidate.exists() for candidate in candidates):
                    violations.append(
                        f"{path.relative_to(ROOT)}:{number}：路径不存在 → `{token}`"
                    )
    return violations


def check_index_completeness() -> list[str]:
    """每份 `.md` 都要出现在索引清单里——新增文档忘登记会被抓到。"""
    if not INDEX.exists():
        return [f"缺少文档索引 {INDEX.relative_to(ROOT)}"]
    text = INDEX.read_text(encoding="utf-8")
    return [
        f"{path.relative_to(DOCS)}：未出现在 {INDEX.name} 的清单里"
        for path in sorted(DOCS.rglob("*.md"))
        if path != INDEX and path.name not in text
    ]


def inventory() -> list[tuple[str, str, int, int]]:
    rows = []
    for path in sorted(DOCS.rglob("*.md")):
        if path == SPEC:
            status = "权威契约"
        elif path == INDEX:
            status = "索引"
        elif is_historical(path):
            status = "历史版本"
        elif is_living(path):
            status = "活文档"
        else:
            status = "现行"
        text = path.read_text(encoding="utf-8")
        rows.append(
            (
                str(path.relative_to(ROOT)).replace("\\", "/"),
                status,
                len(lines_of(path)) - 1,
                len(text.encode("utf-8")),
            )
        )
    return rows


def check() -> list[str]:
    return (
        check_links()
        + check_inline_doc_paths()
        + check_classification()
        + check_values()
        + check_historical_banners()
        + check_citations_of_history()
        + check_parameter_table()
        + check_authority()
        + check_index_completeness()
    )


def main() -> int:
    print("文档守卫")
    print(f"  数值准绳：TIME_FLOOR={TIME_FLOOR}  THETA={THETA}  THETA_HIGH={THETA_HIGH}"
          f"  ADOPTED_SCHEME={ADOPTED_SCHEME}")
    print()
    print(f"  {'路径':<34}{'状态':<10}{'行数':>6}{'字节':>9}")
    for path, status, lines, size in inventory():
        print(f"  {path:<34}{status:<10}{lines:>6}{size:>9}")
    print()

    violations = check()
    if violations:
        print(f"违规 {len(violations)} 项：")
        for line in violations:
            print(f"  ✗ {line}")
        return 1
    print("  ✓ 链接全部可达；文档已分类；现行赋值式与代码一致；"
          "历史版本带状态标注且被引用处已注明；参数表与索引完整")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
