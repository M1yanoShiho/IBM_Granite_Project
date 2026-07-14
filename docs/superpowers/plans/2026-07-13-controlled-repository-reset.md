# 三模块 RAG 干净重建执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 在同一个仓库封存当前旧版本，清空旧主线，从零建立可运行、可替换的 Retriever → Selector → Generator Pipeline，并把 Selector V2 训练计划迁移到新 Selector 文档目录继续设计。

**Architecture:** 旧实现只通过 Git 标签和简短交接文档恢复，不建立 legacy 代码目录。三个模块只传递约定的数据内容：Retriever 交候选证据，Selector 交被选证据 ID、顺序和分数，Generator 交答案与引用；Pipeline 只负责连接和核对 ID，不替任何模块做算法判断。

**Tech Stack:** Python 3.11、Pydantic 2、pytest、Ruff、mypy、Hatchling、GitHub Actions。

## Global Constraints

- 本计划只适用于旧版提交 071e4358dc2e2accc9358b67fc5146539bf790a6；执行前 HEAD 改变就停止并重写计划。
- 旧版标签：legacy-before-three-module-reset-2026-07-13。
- 重建分支：refactor/three-module-baseline。
- 新基线标签：three-module-baseline-v0.1.0。
- 旧代码不迁移、不放进 legacy 目录；需要旧能力时再从标签单独重新实现。
- results/ml_selector 是 Selector V1 的本地实验结果，结论已经写入已跟踪文档；用户已确认原始目录不需要保留，可以在标签验证后删除。
- Selector V1 不进入新 Pipeline。
- Selector V2 是 Selector 模块的 2.0 版本，不是整个系统的版本。
- Selector 跨模块输出只包含所选 evidence ID、rank 和 score；不输出“充分、冲突、缺失”等状态。
- Selector 内部可以使用支持、反驳、来源关系等特征帮助选择，但这些是内部算法，不扩大模块接口。
- Generator 只接收 Pipeline 根据 ID 找回的原始证据；是否回答和怎样组织答案属于 Generator 自己的设计。
- FinanceBench 由团队书面约定保留给完整 RAG 最终对比；不实现程序门禁、权限检查或防绕过代码。
- Selector V2 不使用 FinanceBench 进行训练、开发或测试。
- 共享存储尚未解决；只在文档中标记 Selector V2 训练未开始，不实现复杂存储收据程序。
- 私人用户名、电脑绝对路径、服务器地址、密钥、数据、模型权重和大结果不得进入新主线。
- 删除只在重建分支发生，并且必须先把旧标签推送到 GitHub并验证。
- 禁止 force push、git reset --hard、git add -A 和用 xargs 解析文件名。
- 每次测试失败必须立即停止，不能继续提交。

---

## 0 基础先看这里

### 这次不是“修改旧项目”

这次做的是：

~~~text
当前旧代码
  ↓ 创建标签并推送到 GitHub
GitHub 保存可恢复的旧版本
  ↓ 在新分支工作
删除旧主线文件
  ↓
从零建立三个模块
  ↓
测试通过后推送新分支
  ↓
审核后合并 main
~~~

### 三个模块只交接什么

~~~text
用户问题
  ↓
Retriever
  输出：候选证据的 ID、文字、来源、分数和排名
  ↓
Selector
  输出：选中了哪些 evidence_id、选中顺序和 selection_score
  ↓
Pipeline
  根据 evidence_id 找回 Retriever 的原始文字，防止证据被改写
  ↓
Generator
  输出：答案和使用过的 evidence_id
~~~

Selector 不需要告诉 Generator：

- 证据是否充分；
- 是否存在冲突；
- 缺少什么。

这些如果以后确实需要，必须由团队重新讨论接口版本。本次基线不添加。

### 新目录

~~~text
IBM_Granite_Project/
├── .github/workflows/ci.yml
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── development.md
│   ├── evaluation.md
│   ├── experiments.md
│   ├── handover/legacy-v1.md
│   ├── selector/
│   │   ├── README.md
│   │   ├── TRAINING_PLAN.md
│   │   └── EXPERIMENT_TRACKER.md
│   └── superpowers/plans/2026-07-13-controlled-repository-reset.md
├── src/evidence_rag/
│   ├── contracts/
│   │   ├── models.py
│   │   ├── protocols.py
│   │   └── validation.py
│   ├── retriever/
│   │   ├── chunking.py
│   │   └── bm25.py
│   ├── selector/top_k.py
│   ├── generator/extractive.py
│   ├── pipeline/service.py
│   ├── evaluation/metrics.py
│   ├── cli/smoke.py
│   └── composition.py
├── tests/
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── requirements-dev.lock
└── README.md
~~~

图中省略了每个 Python 包内的 __init__.py。

### 完成标准

1. GitHub 上存在可验证的旧版标签。
2. 新主线没有旧 src、eval、results、scripts、notebooks、proposal 或 app。
3. results/ml_selector 本地原始目录已删除。
4. 新 Pipeline 能运行 Retriever → Selector → Generator。
5. Selector 只能返回 ID、顺序和分数，不能传递或修改证据文字。
6. Pipeline 根据 ID 使用 Retriever 的原始证据。
7. 更换任一模块时，其他模块代码不需要修改。
8. Selector V2 原训练计划和 Tracker 已进入 docs/selector，状态明确是“继续设计，尚未执行”。
9. FinanceBench 规则只作为团队文档约定，不添加程序门禁。
10. 本地测试、类型检查、打包测试和 GitHub CI 全部通过。
11. 新分支审核后进入 origin/main，再创建新基线标签。

---

### Task 1: 先把旧版本保存到 GitHub

**Files:**
- Keep untracked until reset branch: docs/superpowers/plans/2026-07-13-controlled-repository-reset.md
- No old source edits

**Interfaces:**
- Consumes: 当前 week5_MLSelector 分支。
- Produces: GitHub 上可恢复的旧标签和本地重建分支。

- [ ] **Step 1: 在当前仓库根目录确认状态**

Run:

~~~bash
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD
~~~

Expected:

~~~text
## week5_MLSelector...origin/week5_MLSelector
?? docs/superpowers/plans/2026-07-13-controlled-repository-reset.md
071e4358dc2e2accc9358b67fc5146539bf790a6
~~~

第一条命令会显示仓库位置，但不要把显示出来的个人路径复制进文档。

如果还有其他未提交文件，停止。

- [ ] **Step 2: 更新远程信息并确认本地与 GitHub 一致**

Run:

~~~bash
git fetch --prune --tags origin
git rev-list --left-right --count '@{upstream}...HEAD'
~~~

Expected:

~~~text
0	0
~~~

- [ ] **Step 3: 确认新标签和分支名没有被使用**

Run:

~~~bash
git show-ref --verify --quiet refs/tags/legacy-before-three-module-reset-2026-07-13; test $? -eq 1
git show-ref --verify --quiet refs/heads/refactor/three-module-baseline; test $? -eq 1
test -z "$(git ls-remote --tags origin refs/tags/legacy-before-three-module-reset-2026-07-13)"
test -z "$(git ls-remote --heads origin refs/heads/refactor/three-module-baseline)"
~~~

Expected: 没有输出，四项检查成功。

- [ ] **Step 4: 给当前旧提交创建标签并立即推送 GitHub**

Run:

~~~bash
git tag -a legacy-before-three-module-reset-2026-07-13 071e4358dc2e2accc9358b67fc5146539bf790a6 -m "Archive pre-reset pipeline"
git push origin legacy-before-three-module-reset-2026-07-13
git rev-parse 'legacy-before-three-module-reset-2026-07-13^{}'
git ls-remote origin 'refs/tags/legacy-before-three-module-reset-2026-07-13^{}'
~~~

Expected: 本地和远程都显示：

~~~text
071e4358dc2e2accc9358b67fc5146539bf790a6
~~~

标签保存的是已提交文件。results/ml_selector 没有提交，因此不在标签里；用户已确认这个原始结果目录不需要保留。

- [ ] **Step 5: 创建重建分支**

Run:

~~~bash
git switch -c refactor/three-module-baseline legacy-before-three-module-reset-2026-07-13
~~~

Expected: 当前分支变成 refactor/three-module-baseline。

- [ ] **Step 6: 提交计划前检查没有个人绝对路径**

Run:

~~~bash
PRIVATE_MAC_ROOT="/""Users/"
PRIVATE_LINUX_ROOT="/""home/"
PRIVATE_SCRATCH_ROOT="/""scratch/"
if rg -n "$PRIVATE_MAC_ROOT|$PRIVATE_LINUX_ROOT|$PRIVATE_SCRATCH_ROOT" docs/superpowers/plans/2026-07-13-controlled-repository-reset.md; then
  echo "ERROR: personal absolute path found"
  exit 1
fi
~~~

Expected: 没有结果。若出现结果，先删除个人路径。

- [ ] **Step 7: 只提交计划**

Run:

~~~bash
git add docs/superpowers/plans/2026-07-13-controlled-repository-reset.md
git diff --cached --name-status
git commit -m "docs: plan clean three-module rebuild"
~~~

Expected: 暂存列表只有计划文件。

---

### Task 2: 保留 Selector 计划并清空旧主线

**Files:**
- Preserve and move: docs/ml-selector/V2_EXPERIMENT_PLAN.md → docs/selector/TRAINING_PLAN.md
- Preserve and move: docs/ml-selector/V2_EXPERIMENT_TRACKER.md → docs/selector/EXPERIMENT_TRACKER.md
- Delete: 除计划和上面两份 Selector 文档外的全部旧 tracked 文件
- Create: .gitignore
- Create: .python-version
- Create: pyproject.toml
- Create: README.md
- Create: .env.example
- Create: src/evidence_rag package shell

**Interfaces:**
- Consumes: 已验证的旧标签。
- Produces: 没有旧实现的项目外壳，以及仍可继续编辑的 Selector V2 计划。

- [ ] **Step 1: 确认 Selector 计划已经被 Git 保存**

Run:

~~~bash
git cat-file -e 'legacy-before-three-module-reset-2026-07-13:docs/ml-selector/V1_EXPERIMENT_RECORD.md'
git cat-file -e 'legacy-before-three-module-reset-2026-07-13:docs/ml-selector/V2_EXPERIMENT_PLAN.md'
git cat-file -e 'legacy-before-three-module-reset-2026-07-13:docs/ml-selector/V2_EXPERIMENT_TRACKER.md'
~~~

Expected: 三条命令都成功。

V1 结论由旧标签保存，并会写入新的简短 handover。V2 计划和 Tracker 继续留在新主线。

- [ ] **Step 2: 预览 tracked 删除范围**

Run:

~~~bash
git ls-files -- . \
  ':(exclude)docs/superpowers/plans/2026-07-13-controlled-repository-reset.md' \
  ':(exclude)docs/ml-selector/V2_EXPERIMENT_PLAN.md' \
  ':(exclude)docs/ml-selector/V2_EXPERIMENT_TRACKER.md'
~~~

Expected:

- 列出所有旧 tracked 文件。
- 不列出计划、V2 实验计划和 V2 Tracker。
- 中文文件和带空格文件也会正常列出。

- [ ] **Step 3: 删除旧 tracked 树**

Run:

~~~bash
git rm -r -- . \
  ':(exclude)docs/superpowers/plans/2026-07-13-controlled-repository-reset.md' \
  ':(exclude)docs/ml-selector/V2_EXPERIMENT_PLAN.md' \
  ':(exclude)docs/ml-selector/V2_EXPERIMENT_TRACKER.md'
~~~

Expected: 旧文件进入 staged deleted 状态，三份保留文档仍存在。不要运行 git add -A。

- [ ] **Step 4: 把 Selector 计划移动到新模块文档目录**

Run:

~~~bash
mkdir -p docs/selector
git mv docs/ml-selector/V2_EXPERIMENT_PLAN.md docs/selector/TRAINING_PLAN.md
git mv docs/ml-selector/V2_EXPERIMENT_TRACKER.md docs/selector/EXPERIMENT_TRACKER.md
~~~

Expected:

~~~text
docs/selector/TRAINING_PLAN.md
docs/selector/EXPERIMENT_TRACKER.md
~~~

- [ ] **Step 5: 删除已确认不需要的本地 Selector V1 原始结果**

Run:

~~~bash
if test -d results/ml_selector; then
  rm -rf results/ml_selector
fi
test ! -e results/ml_selector
~~~

Expected: 最后一条命令成功。这里不再做逐文件归档，因为用户已经确认整个目录不需要保留，结论由 Git 中的 V1 文档保存。

- [ ] **Step 6: 写入新的忽略规则**

Create .gitignore:

~~~gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
build/
dist/
*.egg-info/

.venv/
.venv-*/
venv/

.env
.env.*
!.env.example

.DS_Store
.idea/
.vscode/
*.swp

/data/
/results/
/runs/
/artifacts/
/outputs/
*.log

/.superpowers/
~~~

- [ ] **Step 7: 写入项目配置**

Create .python-version:

~~~text
3.11
~~~

Create pyproject.toml:

~~~toml
[build-system]
requires = ["hatchling>=1.25,<2"]
build-backend = "hatchling.build"

[project]
name = "evidence-rag"
version = "0.1.0"
description = "Clean Retriever-Selector-Generator pipeline"
requires-python = ">=3.11,<3.12"
dependencies = ["pydantic>=2.7,<3"]

[project.optional-dependencies]
dev = [
  "build>=1.2,<2",
  "mypy>=1.10,<2",
  "pytest>=8,<9",
  "ruff>=0.5,<1",
]

[project.scripts]
evidence-rag-smoke = "evidence_rag.cli.smoke:main"

[tool.hatch.build]
include = ["src/evidence_rag/**"]

[tool.hatch.build.targets.wheel]
packages = ["src/evidence_rag"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src", "tests/typecheck.py"]
~~~

- [ ] **Step 8: 创建最小 Python 包**

Create:

~~~python
# src/evidence_rag/__init__.py
"""Clean Evidence RAG package."""

__version__ = "0.1.0"
~~~

Create each package __init__.py with its matching docstring:

~~~python
# src/evidence_rag/contracts/__init__.py
"""Shared module contracts."""
~~~

~~~python
# src/evidence_rag/retriever/__init__.py
"""Retriever implementations."""
~~~

~~~python
# src/evidence_rag/selector/__init__.py
"""Selector implementations."""
~~~

~~~python
# src/evidence_rag/generator/__init__.py
"""Generator implementations."""
~~~

~~~python
# src/evidence_rag/pipeline/__init__.py
"""Pipeline orchestration."""
~~~

~~~python
# src/evidence_rag/evaluation/__init__.py
"""Evaluation metrics."""
~~~

~~~python
# src/evidence_rag/cli/__init__.py
"""Command-line entry points."""
~~~

Create README.md:

~~~markdown
# Evidence RAG

This repository contains one clean pipeline:

Retriever → Selector → Generator

Start with docs/README.md.

The pre-reset implementation is available from Git tag
legacy-before-three-module-reset-2026-07-13.
~~~

Create .env.example:

~~~dotenv
# Never commit real secrets, usernames, server addresses, or private paths.
GENERATOR_PROVIDER=
GENERATOR_MODEL=
TEAM_ARTIFACT_STORE=
~~~

- [ ] **Step 9: 建立新的 Python 3.11 环境**

Run:

~~~bash
rm -rf .venv
/opt/homebrew/bin/python3.11 -m venv .venv-reset
source .venv-reset/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip freeze --exclude-editable > requirements-dev.lock
python -c "import evidence_rag; print(evidence_rag.__version__)"
~~~

Expected:

~~~text
Python 3.11.x
0.1.0
~~~

- [ ] **Step 10: 检查并提交干净外壳**

Run:

~~~bash
git status --short
git ls-files --others --exclude-standard
git add .gitignore .python-version pyproject.toml requirements-dev.lock README.md .env.example src/evidence_rag
git diff --cached --name-status
git commit -m "chore: replace legacy tree with clean package shell"
~~~

Expected:

- results/ml_selector、.venv 和 .superpowers 不在暂存列表。
- 提交包含旧文件删除、Selector 文档移动和新外壳。

---

### Task 3: 建立只传递内容的模块接口

**Files:**
- Create: src/evidence_rag/contracts/models.py
- Create: src/evidence_rag/contracts/protocols.py
- Create: src/evidence_rag/contracts/validation.py
- Create: tests/contracts/test_models.py
- Create: tests/contracts/test_validation.py

**Interfaces:**
- Consumes: Pydantic。
- Produces: Query、Document、CandidateSet、SelectionResult、SelectedEvidenceSet、GenerationResult 和三个 Protocol。

- [ ] **Step 1: 写失败测试**

Create tests/contracts/test_models.py:

~~~python
import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    SelectionItem,
    SelectionResult,
)


def candidate(evidence_id: str = "ev-1") -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id="doc-1",
        chunk_id=f"chunk-{evidence_id}",
        text="Revenue increased.",
        source_uri="fixture://doc-1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def test_candidate_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError):
        CandidateSet(query_id="q-1", candidates=(candidate(), candidate()))


def test_selector_result_contains_no_evidence_text_or_status() -> None:
    item = SelectionItem(evidence_id="ev-1", selection_score=0.9, selection_rank=1)
    assert not hasattr(item, "text")
    assert not hasattr(item, "status")


def test_selected_ids_must_be_unique() -> None:
    item = SelectionItem(evidence_id="ev-1", selection_score=0.9, selection_rank=1)
    with pytest.raises(ValidationError):
        SelectionResult(query_id="q-1", items=(item, item))


def test_nonempty_answer_requires_citation() -> None:
    with pytest.raises(ValidationError):
        GenerationResult(query_id="q-1", answer="Revenue increased.", cited_evidence_ids=())
~~~

Create tests/contracts/test_validation.py:

~~~python
import pytest

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.contracts.validation import resolve_selection, validate_generation


def candidate(evidence_id: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id="doc-1",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri="fixture://doc-1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def test_pipeline_resolves_original_retriever_text() -> None:
    candidates = CandidateSet(
        query_id="q-1",
        candidates=(candidate("ev-1", "original retriever text"),),
    )
    selection = SelectionResult(
        query_id="q-1",
        items=(SelectionItem(evidence_id="ev-1", selection_score=0.9, selection_rank=1),),
    )
    selected = resolve_selection(candidates, selection)
    assert selected.evidence[0].text == "original retriever text"


def test_unknown_selected_id_is_rejected() -> None:
    candidates = CandidateSet(
        query_id="q-1",
        candidates=(candidate("ev-1", "text"),),
    )
    selection = SelectionResult(
        query_id="q-1",
        items=(SelectionItem(evidence_id="forged", selection_score=0.9, selection_rank=1),),
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        resolve_selection(candidates, selection)


def test_generator_cannot_cite_unselected_id() -> None:
    candidates = CandidateSet(
        query_id="q-1",
        candidates=(candidate("ev-1", "text"),),
    )
    selection = SelectionResult(
        query_id="q-1",
        items=(SelectionItem(evidence_id="ev-1", selection_score=0.9, selection_rank=1),),
    )
    selected = resolve_selection(candidates, selection)
    result = GenerationResult(
        query_id="q-1",
        answer="Unsupported.",
        cited_evidence_ids=("ev-2",),
    )
    with pytest.raises(ValueError, match="unselected evidence"):
        validate_generation(selected, result)
~~~

- [ ] **Step 2: 运行测试并确认实现尚不存在**

Run:

~~~bash
source .venv-reset/bin/activate
pytest tests/contracts -q
~~~

Expected: FAIL，原因是 contracts.models 尚不存在。

- [ ] **Step 3: 实现数据模型**

Create src/evidence_rag/contracts/models.py:

~~~python
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

NonEmpty = Annotated[str, Field(min_length=1)]
PositiveRank = Annotated[int, Field(ge=1)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Query(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    text: NonEmpty


class Document(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    document_id: NonEmpty
    text: NonEmpty
    source_uri: NonEmpty


class EvidenceCandidate(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_id: NonEmpty
    document_id: NonEmpty
    chunk_id: NonEmpty
    text: NonEmpty
    source_uri: NonEmpty
    retrieval_score: FiniteFloat
    retrieval_rank: PositiveRank


class CandidateSet(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    candidates: tuple[EvidenceCandidate, ...]

    @model_validator(mode="after")
    def unique_ids_and_ranks(self) -> "CandidateSet":
        ids = tuple(item.evidence_id for item in self.candidates)
        ranks = tuple(item.retrieval_rank for item in self.candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("candidate evidence IDs must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("candidate retrieval ranks must be unique")
        return self


class SelectionItem(FrozenModel):
    evidence_id: NonEmpty
    selection_score: FiniteFloat
    selection_rank: PositiveRank


class SelectionResult(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    items: tuple[SelectionItem, ...]

    @model_validator(mode="after")
    def unique_ids_and_ranks(self) -> "SelectionResult":
        ids = tuple(item.evidence_id for item in self.items)
        ranks = tuple(item.selection_rank for item in self.items)
        if len(ids) != len(set(ids)):
            raise ValueError("selected evidence IDs must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("selection ranks must be unique")
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("selection items must be ordered by consecutive ranks")
        return self


class SelectedEvidenceSet(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    evidence: tuple[EvidenceCandidate, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> "SelectedEvidenceSet":
        ids = tuple(item.evidence_id for item in self.evidence)
        if len(ids) != len(set(ids)):
            raise ValueError("selected evidence IDs must be unique")
        return self


class GenerationResult(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    answer: str
    cited_evidence_ids: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def answer_and_citations_match(self) -> "GenerationResult":
        if self.answer.strip() and not self.cited_evidence_ids:
            raise ValueError("nonempty answer requires at least one citation")
        if not self.answer.strip() and self.cited_evidence_ids:
            raise ValueError("empty answer cannot contain citations")
        if len(self.cited_evidence_ids) != len(set(self.cited_evidence_ids)):
            raise ValueError("citations must be unique")
        return self
~~~

- [ ] **Step 4: 实现 Protocol**

Create src/evidence_rag/contracts/protocols.py:

~~~python
from typing import Protocol

from evidence_rag.contracts.models import (
    CandidateSet,
    GenerationResult,
    Query,
    SelectedEvidenceSet,
    SelectionResult,
)


class Retriever(Protocol):
    def retrieve(self, query: Query, top_k: int) -> CandidateSet: ...


class Selector(Protocol):
    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult: ...


class Generator(Protocol):
    def generate(
        self,
        query: Query,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult: ...
~~~

- [ ] **Step 5: 实现 Pipeline 边界核对**

Create src/evidence_rag/contracts/validation.py:

~~~python
from evidence_rag.contracts.models import (
    CandidateSet,
    GenerationResult,
    SelectedEvidenceSet,
    SelectionResult,
)


def resolve_selection(
    candidates: CandidateSet,
    selection: SelectionResult,
) -> SelectedEvidenceSet:
    if candidates.query_id != selection.query_id:
        raise ValueError("candidate and selection query IDs differ")
    by_id = {item.evidence_id: item for item in candidates.candidates}
    selected_ids = tuple(item.evidence_id for item in selection.items)
    unknown = tuple(evidence_id for evidence_id in selected_ids if evidence_id not in by_id)
    if unknown:
        raise ValueError(f"selection contains unknown evidence: {unknown}")
    return SelectedEvidenceSet(
        query_id=selection.query_id,
        evidence=tuple(by_id[evidence_id] for evidence_id in selected_ids),
    )


def validate_generation(
    selected: SelectedEvidenceSet,
    result: GenerationResult,
) -> None:
    if selected.query_id != result.query_id:
        raise ValueError("selected evidence and generation query IDs differ")
    allowed = {item.evidence_id for item in selected.evidence}
    unknown = set(result.cited_evidence_ids) - allowed
    if unknown:
        raise ValueError(f"generation cites unselected evidence: {sorted(unknown)}")
~~~

- [ ] **Step 6: 测试、类型检查并提交**

Run:

~~~bash
pytest tests/contracts -q &&
mypy src/evidence_rag/contracts &&
git add src/evidence_rag/contracts tests/contracts &&
git commit -m "feat: define content-only module contracts"
~~~

Expected: 任何测试或类型检查失败时，后面的提交不会执行。

---

### Task 4: 从零实现最小 Retriever

**Files:**
- Create: src/evidence_rag/retriever/chunking.py
- Create: src/evidence_rag/retriever/bm25.py
- Create: tests/retriever/test_bm25.py

**Interfaces:**
- Consumes: Document 和 Query。
- Produces: 有 ID、原文、分数、排名的 CandidateSet。

- [ ] **Step 1: 写 Retriever 测试**

Create tests/retriever/test_bm25.py:

~~~python
from evidence_rag.contracts.models import Document, Query
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.retriever.chunking import WordChunker


def documents() -> tuple[Document, ...]:
    return (
        Document(
            document_id="annual-report",
            text="Revenue increased by ten percent. Operating cost remained stable.",
            source_uri="fixture://annual-report",
        ),
        Document(
            document_id="policy",
            text="The company adopted a new travel policy.",
            source_uri="fixture://policy",
        ),
    )


def test_ids_are_stable_and_have_different_meanings() -> None:
    chunker = WordChunker(chunk_size=5, overlap=1)
    first = chunker.chunk(documents()[0])
    second = chunker.chunk(documents()[0])
    assert first == second
    assert first[0].document_id != first[0].chunk_id
    assert first[0].chunk_id != first[0].evidence_id


def test_bm25_returns_ranked_candidates() -> None:
    retriever = BM25Retriever(documents(), WordChunker(chunk_size=20, overlap=0))
    result = retriever.retrieve(Query(query_id="q-1", text="revenue increase"), top_k=2)
    assert result.candidates[0].document_id == "annual-report"
    assert result.candidates[0].retrieval_rank == 1


def test_no_matching_terms_returns_empty_candidates() -> None:
    retriever = BM25Retriever(documents(), WordChunker(chunk_size=20, overlap=0))
    result = retriever.retrieve(Query(query_id="q-2", text="volcano"), top_k=2)
    assert result.candidates == ()
~~~

- [ ] **Step 2: 确认测试失败**

Run:

~~~bash
pytest tests/retriever -q
~~~

Expected: FAIL，原因是 Retriever 文件尚不存在。

- [ ] **Step 3: 实现切分**

Create src/evidence_rag/retriever/chunking.py:

~~~python
from dataclasses import dataclass
from hashlib import sha256

from evidence_rag.contracts.models import Document


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    evidence_id: str
    text: str
    source_uri: str


class WordChunker:
    version = "word-v1"

    def __init__(self, chunk_size: int = 120, overlap: int = 20) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        words = document.text.split()
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(words), step):
            end = min(start + self.chunk_size, len(words))
            text = " ".join(words[start:end])
            chunk_hash = sha256(
                f"{document.document_id}|{self.version}|{start}|{end}|{text}".encode()
            ).hexdigest()[:16]
            chunk_id = f"chunk-{chunk_hash}"
            evidence_hash = sha256(
                f"{document.source_uri}|{chunk_id}".encode()
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=chunk_id,
                    evidence_id=f"ev-{evidence_hash}",
                    text=text,
                    source_uri=document.source_uri,
                )
            )
            if end == len(words):
                break
        return tuple(chunks)
~~~

- [ ] **Step 4: 实现 BM25**

Create src/evidence_rag/retriever/bm25.py:

~~~python
import math
import re
from collections import Counter
from collections.abc import Iterable

from evidence_rag.contracts.models import CandidateSet, Document, EvidenceCandidate, Query
from evidence_rag.retriever.chunking import Chunk, WordChunker

TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in TOKEN.finditer(text))


class BM25Retriever:
    def __init__(
        self,
        documents: Iterable[Document],
        chunker: WordChunker | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunker = chunker or WordChunker()
        self.k1 = k1
        self.b = b
        self.chunks: tuple[Chunk, ...] = tuple(
            chunk for document in documents for chunk in self.chunker.chunk(document)
        )
        self.tokens = tuple(tokenize(chunk.text) for chunk in self.chunks)
        self.average_length = (
            sum(len(tokens) for tokens in self.tokens) / len(self.tokens)
            if self.tokens
            else 0.0
        )
        self.document_frequency = Counter(
            token for tokens in self.tokens for token in set(tokens)
        )

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored: list[tuple[float, Chunk]] = []
        total = len(self.chunks)
        for chunk, tokens in zip(self.chunks, self.tokens, strict=True):
            counts = Counter(tokens)
            score = 0.0
            for term in tokenize(query.text):
                frequency = counts[term]
                if frequency == 0:
                    continue
                df = self.document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (total - df + 0.5) / (df + 0.5)
                )
                length_ratio = len(tokens) / self.average_length if self.average_length else 0.0
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            if score > 0.0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].evidence_id))
        candidates = tuple(
            EvidenceCandidate(
                evidence_id=chunk.evidence_id,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_uri=chunk.source_uri,
                retrieval_score=score,
                retrieval_rank=rank,
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        )
        return CandidateSet(query_id=query.query_id, candidates=candidates)
~~~

- [ ] **Step 5: 检查并提交**

Run:

~~~bash
pytest tests/retriever -q &&
ruff check src/evidence_rag/retriever tests/retriever &&
mypy src/evidence_rag/retriever &&
git add src/evidence_rag/retriever tests/retriever &&
git commit -m "feat: add clean BM25 retriever baseline"
~~~

Expected: 全部成功后才提交。

---

### Task 5: 从零实现只输出证据 ID 的 Selector

**Files:**
- Create: src/evidence_rag/selector/top_k.py
- Create: tests/selector/test_top_k.py

**Interfaces:**
- Consumes: CandidateSet。
- Produces: SelectionResult，只含 ID、score、rank。

- [ ] **Step 1: 写 Selector 测试**

Create tests/selector/test_top_k.py:

~~~python
from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.selector.top_k import TopKSelector


def candidate(evidence_id: str, score: float, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=f"text {evidence_id}",
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


def test_selector_returns_only_id_score_and_rank() -> None:
    candidates = CandidateSet(
        query_id="q-1",
        candidates=(candidate("ev-low", 0.2, 2), candidate("ev-high", 0.9, 1)),
    )
    result = TopKSelector().select(
        Query(query_id="q-1", text="question"),
        candidates,
        max_selected=1,
    )
    assert result.items[0].evidence_id == "ev-high"
    assert result.items[0].selection_rank == 1
    assert not hasattr(result.items[0], "text")
    assert not hasattr(result, "status")


def test_empty_candidates_return_empty_selection() -> None:
    result = TopKSelector().select(
        Query(query_id="q-2", text="question"),
        CandidateSet(query_id="q-2", candidates=()),
        max_selected=2,
    )
    assert result.items == ()
~~~

- [ ] **Step 2: 确认测试失败**

Run:

~~~bash
pytest tests/selector -q
~~~

Expected: FAIL，原因是 top_k.py 不存在。

- [ ] **Step 3: 实现 Top-K Selector**

Create src/evidence_rag/selector/top_k.py:

~~~python
from evidence_rag.contracts.models import (
    CandidateSet,
    Query,
    SelectionItem,
    SelectionResult,
)


class TopKSelector:
    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected <= 0:
            raise ValueError("max_selected must be positive")
        ranked = sorted(
            candidates.candidates,
            key=lambda item: (-item.retrieval_score, item.evidence_id),
        )
        return SelectionResult(
            query_id=query.query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=item.evidence_id,
                    selection_score=item.retrieval_score,
                    selection_rank=rank,
                )
                for rank, item in enumerate(ranked[:max_selected], start=1)
            ),
        )
~~~

- [ ] **Step 4: 检查并提交**

Run:

~~~bash
pytest tests/selector -q &&
ruff check src/evidence_rag/selector tests/selector &&
mypy src/evidence_rag/selector &&
git add src/evidence_rag/selector tests/selector &&
git commit -m "feat: add content-only selector baseline"
~~~

Expected: 全部成功后才提交。

---

### Task 6: 从零实现最小 Generator

**Files:**
- Create: src/evidence_rag/generator/extractive.py
- Create: tests/generator/test_extractive.py

**Interfaces:**
- Consumes: Pipeline 找回的 SelectedEvidenceSet。
- Produces: answer 和 cited_evidence_ids；不消费 Selector 状态。

- [ ] **Step 1: 写 Generator 测试**

Create tests/generator/test_extractive.py:

~~~python
from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    SelectedEvidenceSet,
)
from evidence_rag.generator.extractive import ExtractiveGenerator


def test_generator_uses_selected_content_and_ids() -> None:
    evidence = EvidenceCandidate(
        evidence_id="ev-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        text="Revenue increased by ten percent.",
        source_uri="fixture://doc-1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )
    result = ExtractiveGenerator().generate(
        Query(query_id="q-1", text="What changed?"),
        SelectedEvidenceSet(query_id="q-1", evidence=(evidence,)),
    )
    assert "Revenue increased" in result.answer
    assert result.cited_evidence_ids == ("ev-1",)


def test_empty_selection_returns_empty_output() -> None:
    result = ExtractiveGenerator().generate(
        Query(query_id="q-2", text="What changed?"),
        SelectedEvidenceSet(query_id="q-2", evidence=()),
    )
    assert result.answer == ""
    assert result.cited_evidence_ids == ()
~~~

- [ ] **Step 2: 确认测试失败**

Run:

~~~bash
pytest tests/generator -q
~~~

Expected: FAIL，原因是 extractive.py 不存在。

- [ ] **Step 3: 实现无外部模型依赖的基线 Generator**

Create src/evidence_rag/generator/extractive.py:

~~~python
from evidence_rag.contracts.models import (
    GenerationResult,
    Query,
    SelectedEvidenceSet,
)


class ExtractiveGenerator:
    def generate(
        self,
        query: Query,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        if query.query_id != selected.query_id:
            raise ValueError("query and selected evidence query IDs differ")
        if not selected.evidence:
            return GenerationResult(
                query_id=query.query_id,
                answer="",
                cited_evidence_ids=(),
            )
        answer = "\n".join(
            f"- {item.text} [{item.evidence_id}]" for item in selected.evidence
        )
        return GenerationResult(
            query_id=query.query_id,
            answer=answer,
            cited_evidence_ids=tuple(item.evidence_id for item in selected.evidence),
        )
~~~

- [ ] **Step 4: 检查并提交**

Run:

~~~bash
pytest tests/generator -q &&
ruff check src/evidence_rag/generator tests/generator &&
mypy src/evidence_rag/generator &&
git add src/evidence_rag/generator tests/generator &&
git commit -m "feat: add minimal generator baseline"
~~~

Expected: 全部成功后才提交。

---

### Task 7: 连接 Pipeline 并证明三个模块可以替换

**Files:**
- Create: src/evidence_rag/pipeline/service.py
- Create: src/evidence_rag/composition.py
- Create: src/evidence_rag/cli/smoke.py
- Create: tests/pipeline/test_service.py
- Create: tests/architecture/test_boundaries.py
- Create: tests/architecture/test_swap_matrix.py
- Create: tests/typecheck.py

**Interfaces:**
- Consumes: 三个 Protocol。
- Produces: EvidenceRAGPipeline.run 和 build_baseline。

- [ ] **Step 1: 写 Pipeline 测试**

Create tests/pipeline/test_service.py:

~~~python
from evidence_rag.composition import build_baseline
from evidence_rag.contracts.models import Document, Query


def test_pipeline_returns_answer_and_original_evidence_id() -> None:
    pipeline = build_baseline(
        (
            Document(
                document_id="doc-1",
                text="Revenue increased by ten percent.",
                source_uri="fixture://doc-1",
            ),
        )
    )
    result = pipeline.run(
        Query(query_id="q-1", text="revenue increase"),
        top_k=3,
        max_selected=2,
    )
    assert result.answer
    assert result.cited_evidence_ids[0].startswith("ev-")


def test_pipeline_returns_empty_output_when_nothing_is_retrieved() -> None:
    pipeline = build_baseline(
        (
            Document(
                document_id="doc-1",
                text="Revenue increased.",
                source_uri="fixture://doc-1",
            ),
        )
    )
    result = pipeline.run(
        Query(query_id="q-2", text="volcano"),
        top_k=3,
        max_selected=2,
    )
    assert result.answer == ""
~~~

- [ ] **Step 2: 确认测试失败**

Run:

~~~bash
pytest tests/pipeline -q
~~~

Expected: FAIL，原因是 Pipeline 和 composition 不存在。

- [ ] **Step 3: 实现只负责连接的 Pipeline**

Create src/evidence_rag/pipeline/service.py:

~~~python
from evidence_rag.contracts.models import GenerationResult, Query
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.contracts.validation import resolve_selection, validate_generation


class EvidenceRAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        selector: Selector,
        generator: Generator,
    ) -> None:
        self.retriever = retriever
        self.selector = selector
        self.generator = generator

    def run(
        self,
        query: Query,
        top_k: int = 20,
        max_selected: int = 10,
    ) -> GenerationResult:
        candidates = self.retriever.retrieve(query, top_k)
        if candidates.query_id != query.query_id:
            raise ValueError("retriever returned the wrong query ID")
        selection = self.selector.select(query, candidates, max_selected)
        if selection.query_id != query.query_id:
            raise ValueError("selector returned the wrong query ID")
        selected = resolve_selection(candidates, selection)
        result = self.generator.generate(query, selected)
        validate_generation(selected, result)
        return result
~~~

Create src/evidence_rag/composition.py:

~~~python
from collections.abc import Iterable

from evidence_rag.contracts.models import Document
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.selector.top_k import TopKSelector


def build_baseline(documents: Iterable[Document]) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=BM25Retriever(documents),
        selector=TopKSelector(),
        generator=ExtractiveGenerator(),
    )
~~~

- [ ] **Step 4: 创建真实 smoke 命令**

Create src/evidence_rag/cli/smoke.py:

~~~python
from evidence_rag.composition import build_baseline
from evidence_rag.contracts.models import Document, Query


def main() -> None:
    pipeline = build_baseline(
        (
            Document(
                document_id="smoke-doc",
                text="The clean pipeline passes evidence between three modules.",
                source_uri="fixture://smoke-doc",
            ),
        )
    )
    result = pipeline.run(
        Query(query_id="smoke-query", text="clean pipeline evidence"),
        top_k=3,
        max_selected=2,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
~~~

- [ ] **Step 5: 写模块边界检查**

Create tests/architecture/test_boundaries.py:

~~~python
import ast
from pathlib import Path

SOURCE = Path("src/evidence_rag")
COMPONENTS = ("retriever", "selector", "generator")


def imports(path: Path) -> tuple[tuple[str, int], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
    return tuple(found)


def test_components_only_import_their_own_package_and_contracts() -> None:
    for component in COMPONENTS:
        allowed = (
            f"evidence_rag.{component}",
            "evidence_rag.contracts",
        )
        for path in (SOURCE / component).rglob("*.py"):
            for module, level in imports(path):
                assert level == 0, f"{path} uses a relative import"
                if module.startswith("evidence_rag"):
                    assert module.startswith(allowed), f"{path} imports {module}"


def test_pipeline_imports_contracts_only() -> None:
    for path in (SOURCE / "pipeline").rglob("*.py"):
        for module, level in imports(path):
            assert level == 0, f"{path} uses a relative import"
            if module.startswith("evidence_rag"):
                assert module.startswith("evidence_rag.contracts"), (
                    f"{path} imports concrete module {module}"
                )
~~~

- [ ] **Step 6: 写模块替换矩阵**

Create tests/architecture/test_swap_matrix.py:

~~~python
from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    Query,
    SelectedEvidenceSet,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.pipeline.service import EvidenceRAGPipeline


def evidence(evidence_id: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


class RetrieverA:
    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        return CandidateSet(query_id=query.query_id, candidates=(evidence("ev-a", "A"),))


class RetrieverB:
    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        return CandidateSet(query_id=query.query_id, candidates=(evidence("ev-b", "B"),))


class SelectorA:
    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        item = candidates.candidates[0]
        return SelectionResult(
            query_id=query.query_id,
            items=(SelectionItem(evidence_id=item.evidence_id, selection_score=1.0, selection_rank=1),),
        )


class SelectorB:
    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        item = candidates.candidates[0]
        return SelectionResult(
            query_id=query.query_id,
            items=(SelectionItem(evidence_id=item.evidence_id, selection_score=0.5, selection_rank=1),),
        )


class GeneratorA:
    def generate(
        self,
        query: Query,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        item = selected.evidence[0]
        return GenerationResult(
            query_id=query.query_id,
            answer=f"A:{item.text}",
            cited_evidence_ids=(item.evidence_id,),
        )


class GeneratorB:
    def generate(
        self,
        query: Query,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        item = selected.evidence[0]
        return GenerationResult(
            query_id=query.query_id,
            answer=f"B:{item.text}",
            cited_evidence_ids=(item.evidence_id,),
        )


def run(
    retriever: Retriever,
    selector: Selector,
    generator: Generator,
) -> GenerationResult:
    return EvidenceRAGPipeline(retriever, selector, generator).run(
        Query(query_id="q", text="question"),
        top_k=1,
        max_selected=1,
    )


def test_retriever_can_be_replaced_alone() -> None:
    assert run(RetrieverA(), SelectorA(), GeneratorA()).answer == "A:A"
    assert run(RetrieverB(), SelectorA(), GeneratorA()).answer == "A:B"


def test_selector_can_be_replaced_alone() -> None:
    assert run(RetrieverA(), SelectorA(), GeneratorA()).answer == "A:A"
    assert run(RetrieverA(), SelectorB(), GeneratorA()).answer == "A:A"


def test_generator_can_be_replaced_alone() -> None:
    assert run(RetrieverA(), SelectorA(), GeneratorA()).answer == "A:A"
    assert run(RetrieverA(), SelectorA(), GeneratorB()).answer == "B:A"
~~~

- [ ] **Step 7: 写静态签名检查**

Create tests/typecheck.py:

~~~python
from evidence_rag.contracts.models import Document
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.selector.top_k import TopKSelector

documents = (
    Document(document_id="doc", text="text", source_uri="fixture://doc"),
)
retriever: Retriever = BM25Retriever(documents)
selector: Selector = TopKSelector()
generator: Generator = ExtractiveGenerator()
~~~

- [ ] **Step 8: 运行全部连接测试并提交**

Run:

~~~bash
pytest tests/pipeline tests/architecture -q &&
python -m evidence_rag.cli.smoke &&
mypy src tests/typecheck.py &&
ruff check src tests &&
git add src/evidence_rag/pipeline src/evidence_rag/composition.py src/evidence_rag/cli tests/pipeline tests/architecture tests/typecheck.py &&
git commit -m "feat: connect and verify replaceable modules"
~~~

Expected: 三个替换测试和 smoke 全部通过后才提交。

---

### Task 8: 建立模块评估与清晰交接文档

**Files:**
- Create: src/evidence_rag/evaluation/metrics.py
- Create: tests/evaluation/test_metrics.py
- Create: docs/README.md
- Create: docs/architecture.md
- Create: docs/development.md
- Create: docs/evaluation.md
- Create: docs/experiments.md
- Create: docs/handover/legacy-v1.md
- Create: docs/selector/README.md
- Modify: docs/selector/TRAINING_PLAN.md
- Modify: docs/selector/EXPERIMENT_TRACKER.md
- Create: .github/workflows/ci.yml

**Interfaces:**
- Consumes: 新 Pipeline、旧 V1 结论文档、已迁移 V2 计划。
- Produces: 团队开发规则、Selector 后续入口、最小指标和 CI。

- [ ] **Step 1: 实现三个模块都能使用的简单指标**

Create src/evidence_rag/evaluation/metrics.py:

~~~python
def evidence_recall(
    predicted_ids: tuple[str, ...],
    required_ids: tuple[str, ...],
) -> float:
    if not required_ids:
        raise ValueError("required_ids must not be empty")
    return len(set(predicted_ids) & set(required_ids)) / len(set(required_ids))


def evidence_precision(
    predicted_ids: tuple[str, ...],
    relevant_ids: tuple[str, ...],
) -> float:
    if not predicted_ids:
        return 0.0
    return len(set(predicted_ids) & set(relevant_ids)) / len(set(predicted_ids))


def citation_validity(
    cited_ids: tuple[str, ...],
    selected_ids: tuple[str, ...],
) -> float:
    if not cited_ids:
        return 0.0
    allowed = set(selected_ids)
    return sum(evidence_id in allowed for evidence_id in cited_ids) / len(cited_ids)
~~~

Create tests/evaluation/test_metrics.py:

~~~python
import pytest

from evidence_rag.evaluation.metrics import (
    citation_validity,
    evidence_precision,
    evidence_recall,
)


def test_recall_measures_required_evidence_kept() -> None:
    assert evidence_recall(("ev-1", "ev-2"), ("ev-2", "ev-3")) == 0.5


def test_precision_measures_selected_noise() -> None:
    assert evidence_precision(("ev-1", "ev-bad"), ("ev-1",)) == 0.5


def test_empty_required_set_is_not_silently_scored() -> None:
    with pytest.raises(ValueError):
        evidence_recall(("ev-1",), ())


def test_citation_validity_uses_selected_ids() -> None:
    assert citation_validity(("ev-1", "ev-bad"), ("ev-1",)) == 0.5
~~~

- [ ] **Step 2: 写项目文档入口**

Create docs/README.md:

~~~markdown
# Project guide

Read in this order:

1. architecture.md — what each module receives and returns;
2. development.md — how module teams work independently;
3. evaluation.md — module tests and final complete-RAG evaluation;
4. experiments.md — how plans, results, and large artifacts are recorded;
5. selector/README.md — current Selector status and next design work;
6. handover/legacy-v1.md — what the archived project tried.

Old implementation details are available from Git tag
legacy-before-three-module-reset-2026-07-13.
~~~

Create docs/architecture.md:

~~~markdown
# Architecture

The only production flow is:

Query → Retriever → CandidateSet → Selector → SelectionResult
→ Pipeline resolves canonical evidence → Generator → GenerationResult.

## Retriever

Returns candidate evidence text, source, score, and rank.

## Selector

Returns selected evidence IDs, selection scores, and selection ranks.
It does not return evidence text, sufficiency status, missing facts, or conflict status.
Selector algorithms may use any approved internal features, but the external output stays small.

## Pipeline

Connects modules and resolves selected IDs back to the exact Retriever candidates.
It contains no retrieval, selection, or generation algorithm.

## Generator

Receives the selected canonical evidence and returns an answer plus cited evidence IDs.
Its answering, refusal, and uncertainty behavior belongs to Generator development.
~~~

Create docs/development.md:

~~~markdown
# Parallel development

## Ownership

- Retriever team: src/evidence_rag/retriever and tests/retriever.
- Selector team: src/evidence_rag/selector, tests/selector, and docs/selector.
- Generator team: src/evidence_rag/generator and tests/generator.
- Integration owner: contracts, pipeline, composition, and architecture tests.

## Merge rule

A module improvement may be connected to the baseline only when:

1. it keeps the current Protocol;
2. its own tests pass;
3. the module swap tests pass;
4. the complete Pipeline runs;
5. both module metrics and complete-Pipeline metrics are reported.

Contract changes require agreement from all three module teams.
~~~

- [ ] **Step 3: 写 FinanceBench 和评估约定，不实现代码门禁**

Create docs/evaluation.md:

~~~markdown
# Evaluation agreement

## Module development

Each module team may choose datasets appropriate to its own objective.
The chosen training, development, and test roles must be written in that module's plan.

## Complete-Pipeline development

The team uses agreed development data to connect Retriever, Selector, and Generator.
These results are for development, not the final comparison.

## FinanceBench

The team agrees that FinanceBench is reserved for final comparison of the complete RAG
system against other RAG systems.

It will not be used for Retriever, Selector, or Generator training, development, tuning,
or process testing. This is a team workflow agreement; the code does not implement a
FinanceBench access restriction.

The other two possible final benchmarks have not been selected.
~~~

- [ ] **Step 4: 写实验结果保存规则**

Create docs/experiments.md:

~~~markdown
# Experiment records and artifacts

## Git stores

- experiment plan;
- experiment tracker;
- Git commit and configuration;
- compact aggregate and per-query results needed to understand the conclusion;
- interpretation, limitations, and next decision.

## Shared artifact storage stores

- datasets;
- feature caches;
- checkpoints;
- model weights;
- full predictions;
- large logs and plots.

## Current storage status

The shared artifact location has not been confirmed. Selector V2 training must not start
until at least two team members can access the same location and one member can read a
checkpoint written by another.

Do not make a private server path the only experiment record.
~~~

- [ ] **Step 5: 写旧工作简洁交接**

Create docs/handover/legacy-v1.md:

~~~markdown
# Legacy V1 handover

## Recovery

- Tag: legacy-before-three-module-reset-2026-07-13
- Commit: 071e4358dc2e2accc9358b67fc5146539bf790a6
- Temporary recovery:
  git worktree add ../IBM_Granite_Project_legacy legacy-before-three-module-reset-2026-07-13
- One-file recovery example:
  git show legacy-before-three-module-reset-2026-07-13:src/rag_pipeline.py

## What was done

The archived repository explored dense and hybrid retrieval, Query2Doc, NIAH and
artificial counterfactual construction, corroboration reranking, Astute RAG, and ML
Selector V1.

Selector V1 used fixed Top-20 candidates and selected Top-10 evidence. It tested
relevance, rank, support, source, condition, and learned reranking signals.

## What the work showed

- Some approaches improved whether required evidence entered the final context window.
- Those gains did not prove that harmful or misleading evidence was reliably removed.
- Selector V1 did not pass its full promotion gates and was not connected as the new
  production Selector.
- Artificial-needle and counterfactual construction reflected one diagnostic setup,
  not the final task definition.
- Module responsibilities and interfaces were mixed, motivating the clean reset.

## Result disposition

The compact conclusions were already recorded in the old tracked documentation and are
recoverable from the tag. The local results/ml_selector directory contained the full
non-final experiment output and was intentionally deleted during the reset.

Do not present V1 diagnostics as results of the new Pipeline.
~~~

- [ ] **Step 6: 建立 Selector 新文档入口**

Create docs/selector/README.md:

~~~markdown
# Selector module

## Module contract

Input: Query plus CandidateSet from Retriever.

Output: selected evidence IDs, selection scores, and selection ranks.

The Pipeline resolves those IDs to the original Retriever evidence before Generator use.
The interface does not include sufficiency, missing-fact, or conflict status.

## Current implementation

src/evidence_rag/selector/top_k.py is only a replaceable baseline.

## Selector V2 status

TRAINING_PLAN.md and EXPERIMENT_TRACKER.md were migrated from the old project so work
can continue after the reset.

They are planning inputs, not an approved command to start training. Before execution,
the Selector team must re-check:

1. whether the task and labels still match the new project definition;
2. whether old NIAH, artificial-needle, and counterfactual construction should be used;
3. the training, development, and test datasets;
4. the baseline and success metrics;
5. the team-accessible artifact location;
6. that FinanceBench is excluded from Selector work by team agreement.

Training remains not started until those decisions are reviewed.
~~~

- [ ] **Step 7: 将已迁移的 V2 计划明确标成继续设计的草稿**

At the top of docs/selector/TRAINING_PLAN.md, immediately after the title, insert exactly:

~~~markdown
> **清理迁移状态：待重新设计，不得直接开始训练。**
>
> 这份文件从旧项目迁移，用于保留已经思考过的 Graph Selector 2.0 方案。
> 旧项目的 NIAH、人工假针、counterfactual、标签和评估假设不自动成为新项目方案。
> 清理完成后，Selector 小组要先重新确认任务、数据、标签、指标和共享存储。
> 当前唯一确定的跨模块接口是：候选证据集合 → 排序后的 selected evidence IDs。
> FinanceBench 由团队约定保留给完整 RAG 最终对比，不用于 Selector V2。
~~~

Replace:

~~~markdown
**状态：** 待执行
~~~

with:

~~~markdown
**状态：** 迁移后的设计草稿；尚未批准执行
~~~

Replace the original data-boundary line with:

~~~markdown
**数据边界：** FinanceBench 不属于 Selector V2 的训练、开发或测试数据；本规则由团队文档约定。
~~~

Replace the original V1/V2 FinanceBench blockquote with:

~~~markdown
> **V1/V2 边界：** V1 是历史探索，V2 从新的任务和数据设计重新确认。FinanceBench 只属于完整 RAG 的最终系统对比，不参与本 Selector 计划。
~~~

Replace the old section heading and first paragraph:

~~~markdown
## 0. Design review status

本计划经过三轮独立只读设计审核，最终 verdict 为 **PASS**。PASS 只表示协议内部一致、可审计且能在 zero-new-human-annotation 条件下执行，不保证实现正确或结果一定支持主张。
~~~

with:

~~~markdown
## 0. 旧计划审核记录

旧版本曾检查过这份方案的内部一致性，但该检查不等于新项目已经批准执行。
清理后必须按照新的模块定义重新审核任务、数据构造、标签和实验条件。
~~~

Replace the paragraph that says the V1 FinanceBench output is \`EXPOSED_DIAGNOSTIC_ONLY\` with:

~~~markdown
FinanceBench 不属于 Selector V2 的训练、开发、调参、测试或失败分析数据。团队将它保留给完整 RAG 的最终系统对比。
~~~

Replace the Tracker plan link in docs/selector/EXPERIMENT_TRACKER.md:

~~~markdown
**Plan:** [TRAINING_PLAN.md](TRAINING_PLAN.md)
~~~

Replace the Tracker protocol line with:

~~~markdown
**Protocol:** 当前为迁移草稿；FinanceBench 不用于 Selector V2
~~~

- [ ] **Step 8: 创建 CI**

Create .github/workflows/ci.yml:

~~~yaml
name: ci

on:
  pull_request:
  push:
    branches: [main, refactor/three-module-baseline]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -r requirements-dev.lock
      - run: python -m pip install --no-deps -e .
      - run: pytest
      - run: ruff check src tests
      - run: mypy src tests/typecheck.py
      - run: python -m build
      - run: python -m evidence_rag.cli.smoke
~~~

- [ ] **Step 9: 运行全部本地检查**

Run:

~~~bash
pytest &&
ruff check src tests &&
mypy src tests/typecheck.py &&
python -m build &&
python -m evidence_rag.cli.smoke
~~~

Expected: 全部成功。

- [ ] **Step 10: 检查不存在私人路径或旧结果依赖**

Run:

~~~bash
PRIVATE_MAC_ROOT="/""Users/"
PRIVATE_LINUX_ROOT="/""home/"
PRIVATE_SCRATCH_ROOT="/""scratch/"
SSH_SCHEME="ssh""://"
if rg -n "$PRIVATE_MAC_ROOT|$PRIVATE_LINUX_ROOT|$PRIVATE_SCRATCH_ROOT|$SSH_SCHEME" . \
  --glob '!.git/**' \
  --glob '!.venv-reset/**' \
  --glob '!.superpowers/**'; then
  echo "ERROR: private infrastructure reference found"
  exit 1
fi
if rg -n "results/ml_selector" src tests README.md docs \
  --glob '!docs/handover/legacy-v1.md' \
  --glob '!docs/superpowers/plans/**'; then
  echo "ERROR: deleted result directory is still an active dependency"
  exit 1
fi
~~~

Expected:

- 第一条没有私人地址结果。
- 第二条只允许旧交接和本执行计划提及已删除目录。

- [ ] **Step 11: 提交指标、文档和 CI**

Run:

~~~bash
git add src/evidence_rag/evaluation tests/evaluation docs .github/workflows/ci.yml &&
git diff --cached --name-status &&
git commit -m "docs: establish module workflow and selector handover"
~~~

Expected: 提交中没有数据、模型、原始结果或私人路径。

---

### Task 9: 最终测试、合并 main 和创建新标签

**Files:**
- No new source files

**Interfaces:**
- Consumes: 干净且通过测试的重建分支。
- Produces: origin/main 上的新基线和 GitHub 新标签。

- [ ] **Step 1: 确认本地工作区和旧标签**

Run:

~~~bash
git status --short --branch
git rev-parse 'legacy-before-three-module-reset-2026-07-13^{}'
git ls-remote origin 'refs/tags/legacy-before-three-module-reset-2026-07-13^{}'
test ! -e results/ml_selector
~~~

Expected:

- 工作区干净。
- 旧标签本地和远程都指向 071e4358dc2e2accc9358b67fc5146539bf790a6。
- results/ml_selector 不存在。

- [ ] **Step 2: 最后一次本地验收**

Run:

~~~bash
source .venv-reset/bin/activate
pytest &&
ruff check src tests &&
mypy src tests/typecheck.py &&
rm -rf dist build &&
python -m build &&
python -m evidence_rag.cli.smoke
~~~

Expected: 全部成功。

- [ ] **Step 3: 推送重建分支**

Run:

~~~bash
git push -u origin refactor/three-module-baseline
~~~

Expected: GitHub 创建同名分支。禁止 force。

- [ ] **Step 4: 创建 PR**

Run:

~~~bash
gh pr create --base main --head refactor/three-module-baseline --title "Rebuild clean Retriever-Selector-Generator pipeline" --body "Archives the old implementation by tag, rebuilds a content-only three-module baseline, and carries the Selector V2 planning documents forward for redesign."
~~~

Expected: 输出 PR 链接。

- [ ] **Step 5: 等待这个 PR 自己的检查**

Run:

~~~bash
gh pr checks --watch
gh pr view --json state,mergeable,reviewDecision,statusCheckRollup
~~~

Expected:

- 所有检查 SUCCESS。
- PR 可以合并。
- 审核满足团队规则。

- [ ] **Step 6: 合并到 main**

Run:

~~~bash
REVIEWED_COMMIT="$(git rev-parse HEAD)"
gh pr merge --merge --delete-branch
git fetch --prune origin
git switch main
git pull --ff-only origin main
git merge-base --is-ancestor "$REVIEWED_COMMIT" origin/main
~~~

Expected: 最后一条成功，证明 main 包含经过审核的新版本。

如果 GitHub 要求其他成员点击合并，就等待合并后再运行 fetch、switch、pull 和检查。

- [ ] **Step 7: 查看 main 最新提交自己的 CI**

Run:

~~~bash
MAIN_COMMIT="$(git rev-parse origin/main)"
gh run list --branch main --commit "$MAIN_COMMIT" --limit 1
MAIN_RUN_ID="$(gh run list --branch main --commit "$MAIN_COMMIT" --limit 1 --json databaseId --jq '.[0].databaseId')"
test -n "$MAIN_RUN_ID"
gh run watch "$MAIN_RUN_ID" --exit-status
~~~

Expected: 与 origin/main 精确 commit 对应的工作流成功。

- [ ] **Step 8: 创建新基线标签**

Run:

~~~bash
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
git tag -a three-module-baseline-v0.1.0 -m "Clean three-module RAG baseline"
git push origin three-module-baseline-v0.1.0
git rev-parse 'three-module-baseline-v0.1.0^{}'
git ls-remote origin 'refs/tags/three-module-baseline-v0.1.0^{}'
~~~

Expected: 本地和远程新标签 hash 一致。

---

## 红灯停止条件

1. 执行前 HEAD 不是 071e4358dc2e2accc9358b67fc5146539bf790a6。
2. 除本计划外还有其他普通未提交文件。
3. 本地与 origin/week5_MLSelector 不同步。
4. 旧标签没有成功推送或远程 hash 不一致。
5. 删除预览包含本计划、V2 TRAINING_PLAN 或 V2 TRACKER。
6. 暂存列表出现 results/ml_selector、.env、服务器地址、数据或模型。
7. 新 Selector 接口出现 evidence text、sufficiency、missing facts 或 conflict status。
8. Selector V2 训练计划没有迁移到 docs/selector。
9. 测试、类型检查、模块替换、build 或 CI 失败。
10. PR 未合并到 main 就准备创建新基线标签。
11. 共享存储未确认却准备启动 Selector V2 训练。

## 本计划不做

- 不迁移旧代码。
- 不保留 results/ml_selector 原始结果目录。
- 不重新包装 Selector V1 结果。
- 不启动 Selector V2 训练。
- 不确认旧 Graph/NIAH/counterfactual 设计仍然正确。
- 不决定 Selector V2 最终数据、标签、模型和指标。
- 不实现 FinanceBench 程序门禁。
- 不连接私人服务器。
- 不实现生产级 Granite Generator。

## 清理完成后的下一项工作

清理完成后，Selector 小组从 docs/selector/README.md 开始，重新审查
TRAINING_PLAN.md。第一项不是运行训练，而是确认：

1. Selector 的确切任务仍是 CandidateSet → ranked selected evidence IDs；
2. 哪些旧数据构造假设需要删除；
3. 使用什么训练、开发和模块测试数据；
4. 主要 baseline、指标和成功条件；
5. 结果写入哪个团队可访问的位置。

确认这些内容后，再为 Selector V2 单独写一份可执行训练计划。
