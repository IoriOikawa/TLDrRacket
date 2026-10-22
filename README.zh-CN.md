# TLDrRacket

![TLDrRacket — CS4400 / CS5400 课程阅读助手](docs/assets/readme-header.png)

[English](README.md) · **简体中文** · [日本語](README.ja.md)

一个为东北大学 [CS4400 / CS5400：编程语言原理](https://pl.barzilay.org/)课程设计的轻量阅读助手。用自然语言提问，即可获得以已索引阅读材料为依据的回答，并打开最多 **五处相关的阅读位置**。匹配较少时，展示的来源也会相应减少；没有匹配时，系统会明确说明无法作答，且不会调用生成模型。

应用使用 Python、Flask、本地 Chroma、Gemini，以及现有的 HTML/CSS/JavaScript 前端。无需前端构建步骤、独立数据库服务、智能体框架或任务队列。本项目也希望为构建 RAG 应用提供一个易于上手的示例。

## 快速开始

请使用 Python **3.12**。以下命令均在仓库根目录运行。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，设置 `GOOGLE_API_KEY`。该文件使用标准的 `KEY=value` 格式；如果旧文件中只有裸 API 密钥，需要先转换为这一格式。进程中已有的环境变量优先于文件中的设置。切勿将密钥放入前端代码。

```sh
python index.py build
python webserver.py
```

打开 [localhost:5000](http://127.0.0.1:5000)。第一条命令会使用嵌入 API 构建持久化索引；服务器不会自动构建索引。没有索引时，页面和 `/healthz` 仍可访问，但提问需要已配置的密钥和就绪的索引。有关 Gunicorn、systemd、Nginx 和 HTTPS 的部署，请参阅 [Linux VPS 指南](docs/deployment.md)。

## RAG 的工作方式

```text
PDF/文本 → 按页提取 → 重叠分块 → 嵌入向量 → 本地 Chroma
问题 → 一次向量检索 → 距离过滤 → 按页去重 → 最多 5 段摘录
                                                   ↓
                             Gemini 回答 + 对应的阅读位置
```

1. `index.py` 读取 `DATA_PATH` 下的 PDF 和 UTF-8 编码的 `.txt` 文件。PDF 页面会被切分为每块 800 个字符、相邻块重叠 80 个字符的文本块。空白页会被跳过；扫描版 PDF 需要先进行 OCR，才能导入。
2. Chroma 存储嵌入向量及元数据：相对文件名、从零开始的 PDF 页码、文本块编号，以及包含内容哈希的稳定 ID。集合明确使用**余弦距离**。
3. `rag.py` 一次检索最多 20 个候选文本块。随后按距离过滤，每个文件的每一页只保留最佳文本块，最终最多选取五个。这代表五处**位置**，不一定来自五份不同的文档。
4. 只有选中的文本块会进入生成提示词。这些相同的文本块也用于提供来源链接和预览。模型会被要求使用 `[1]`、`[2]` 等编号引用，并在证据不足时明确说明。
5. 没有文本块通过筛选时，服务器会返回 `status: "no_match"`、说明文字和 `sources: []`，且不会调用生成模型。问题的向量化仍然需要调用嵌入 API。

来源响应除了简短的 `preview`，还通过 `content` 提供完整的检索摘录，并保留其中的换行和缩进。对于 `data/5400Notes.txt` 等 `.txt` 笔记，`format` 为 `markdown`；PDF 摘录则使用 `text`。来源下方的 **Read excerpt** 展示原始段落，并按相应格式显示标题、列表、表格和代码块。Markdown 摘录默认展开，可点击收起；PDF 的纯文本摘录则可按需展开。这就是生成回答时使用的同一段检索内容，并非整个文件，也不是模型额外生成的摘要。由于索引存储的是长度受限的文本块，摘录可能从某个 Markdown 结构的中间开始或结束。

界面会立即显示等待动画和已用时间，阻止重复提交，并在请求成功、失败或达到 90 秒超时后恢复输入。PDF 链接使用**从 1 开始的 PDF 页码**，可能与页面上印刷的页码不同。文本来源会打开原始文本文件；可展开的摘录则便于在聊天中阅读带格式的内容。生成的 Markdown 会经过安全清理，用户问题则作为纯文本呈现。

## 配置

参阅 [`.env.example`](.env.example)。路径可以使用绝对路径；相对路径以进程的工作目录为基准解析。

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `GOOGLE_API_KEY` | 必填 | 服务器端 Gemini Developer API 密钥 |
| `CHAT_MODEL` | `gemini-3.5-flash-lite` | 用于生成回答的模型 |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | 索引和查询共用的嵌入模型 |
| `EMBEDDING_DIMENSIONS` | `768` | 嵌入向量维度 |
| `DATA_PATH` | 仓库中的 `data/` | PDF 和文本阅读材料目录 |
| `CHROMA_PATH` | 仓库中的 `chroma/` | 持久化本地索引目录 |
| `MAX_DISTANCE` | `0.35` | **暂定的**最大可接受余弦距离 |
| `CANDIDATE_COUNT` | `20` | 过滤和去重前的候选文本块数量；范围为 5–100 |
| `PROVIDER_TIMEOUT` | `30` | 模型服务传输超时，单位为秒；最大值为 45 |
| `MAX_MESSAGE_LENGTH` | `2000` | 非空问题的最大长度 |

默认设置面向 Gemini Developer API，而非 Vertex AI。历史 notebook 中使用的 Gemini 1.5 Flash 和 `text-embedding-004` 标识符均已停用；请参阅 Google 的[版本说明](https://ai.google.dev/gemini-api/docs/changelog)和[弃用时间表](https://ai.google.dev/gemini-api/docs/deprecations)。默认文本嵌入模型仍受支持，也是本项目已进行集成测试的模型。更改嵌入模型、维度或分块设置后，需要显式重建索引；仅更改对话模型或距离阈值则不需要。

### 调整相关性，而非凑满来源数量

**余弦距离越小，匹配越接近。** `0.35` 只是一个起始值，并非测得的置信度，也不是普遍适用的正确阈值。在当前检出的代码版本中，尚未使用真实的 Gemini 嵌入向量，针对本课程语料校准该值。

可以检查候选结果，而无需支付回答生成的费用：

```sh
python index.py probe "How do closures capture an environment?"
python index.py probe "What is the weather tomorrow?"
python index.py calibrate docs/retrieval-eval.json
```

`probe` 会输出候选结果的距离、预览，以及各结果是否通过阈值；按页去重和最多五处位置的限制在之后执行。`calibrate` 对每个已标注的问题只生成一次嵌入向量，然后遍历 0.05 到 0.80 的阈值。它会报告正确匹配或漏掉的课程问题、匹配到错误来源的回答，以及本会被接受的无关问题。整个过程不会生成回答。

请根据实际阅读材料，检查并扩充 `docs/retrieval-eval.json` 中的初始标注，也应纳入教材等有效的替代来源。选择阈值时，应尽量避免接受无关内容和错误来源，同时仍能找到课程问题的相关材料。再使用独立留出的测试问题进行检查；如有需要，也应加入中文问题。之后设置 `MAX_DISTANCE` 并重启服务。此辅助工具只检查是否检索到了至少一个预期来源，不会评价每个展示的来源或生成的回答。

### 更新材料或迁移索引

先停止 Web 服务，再运行：

```sh
python index.py build
```

未变更的文本块会被复用；变更或新增的文本块会重新生成嵌入向量；已删除或编辑的阅读材料所留下的过时文本块会被移除。整个 `DATA_PATH` 目录都应视为目标语料库：从该目录删除文件，会在下次构建时删除它们对应的索引文本块。

旧 notebook 创建的索引，以及更改过嵌入设置的索引，都需要迁移。请先备份索引，再运行：

```sh
python index.py build --rebuild
```

这只会替换应用的 `tldrracket` 集合。如果索引是在加入 Markdown 空白保留功能之前构建的，请运行一次 `build --rebuild`，以完整保留缩进代码。写入中断或失败会留下 `.building` 标记，防止服务使用不完整的索引；请参阅[恢复步骤](docs/deployment.md#recover-an-interrupted-build)。每次成功更新后都应重启服务器。恢复备份时，请确保数据、索引和模型配置相互配套。

## HTTP API

向 `POST /api/ask` 发送请求，并设置 `Content-Type: application/json`：

```json
{"message": "What is lexical scope?"}
```

成功响应包含 `response`、`status`（`ok` 或 `no_match`）和 `sources`。每个来源包含 `id`、`file`、`page`（PDF 使用从 1 开始的页码，文本为 `null`）、`url`、`preview`、`content`（完整的检索文本块）、`format`（`.txt` 为 `markdown`，PDF 为 `text`）和 `distance`。距离仅用于诊断，并非置信度分数。

| 端点 / 状态码 | 含义 |
| --- | --- |
| `GET /healthz` | 进程存活；不调用模型 |
| `GET /readyz` | 密钥已配置，且可以打开兼容的非空索引；不调用模型 |
| `GET /documents/<filename>` | 提供 `DATA_PATH` 内的 PDF 或文本阅读材料 |
| `400`、`413` | 问题无效，或请求体过大 |
| `429` | 当前进程中已有两个问题正在处理，或触发代理限流 |
| `502`、`504` | 模型服务失败、回答为空，或检测到模型服务超时 |
| `503` | 索引或配置缺失、不兼容、为空，或正在更新 |

错误使用 `{"error": "..."}` 格式。API 响应不会被缓存。Nginx 可能返回自己的错误页面；界面也能处理非 JSON 格式的错误。就绪检查不会验证远程 API 密钥、配额或模型可用性。浏览器超时会停止等待，但不保证取消服务器上已经运行的模型服务请求。

## 文件与学习路径

- `index.py`：文档加载、分块、增量索引、候选结果探查和阈值遍历。
- `rag.py`：共享配置、索引兼容性、检索、来源和回答生成。
- `webserver.py`：HTTP 验证、延迟初始化、并发上限和安全的文档访问。
- `static/index.html`、`static/chat.js`、`static/style.css`：现有前端，包含等待反馈和来源链接。
- `deploy/`：生产环境进程和反向代理配置示例。
- `TLDrRacket.ipynb`、`evaluation.ipynb`：保留供参考的历史实验。它们不是当前受支持的运行方式，复用前需要更新依赖和模型。旧版评估使用来源 ID 作为上下文；有意义的忠实度评估应使用实际摘录文本。

## 验证

运行时测试使用 Python 标准库，无需测试框架或 API 密钥：

```sh
python -m unittest discover -s tests -v
```

测试使用真实的临时 Chroma 集合、确定性的**本地测试向量**，以及隔离的模型服务替身。覆盖范围包括余弦距离过滤、0–5 个结果、来源与上下文的一致性、去重、索引更新和迁移、请求验证、失败场景以及路径遍历。它们不能证明真实模型的回答质量。

可选的浏览器检查只要求在开发环境中安装 Playwright。先在另一个终端启动 `python webserver.py`，然后运行：

```sh
python -m pip install playwright
python -m playwright install chromium
python tests/browser_smoke.py
```

这些检查通过拦截响应来验证等待状态、重复提交、0/1/5 个来源、超时和错误后的恢复、Markdown 回答与来源摘录、安全清理，以及移动端行为。

## 范围与限制

这是一个单节点 RAG 示例，最多同时处理两个问题，并使用持久化的嵌入式数据库。检索是近似的；距离阈值仍可能接受无关段落，或排除有用内容。提示词会引导模型依据来源回答并使用编号引用，但不会正式验证论断或引用。项目不提供网页浏览、重排序模型、聊天记忆、OCR 服务或自动语料同步。阅读材料的覆盖范围取决于你建立索引的文件，不一定与当前课程网站一致。文档路由会向网站访客开放 `DATA_PATH` 中的阅读材料，因此请只使用你打算通过此工具公开的语料。

JavaScript 依赖随项目收录在 `static/vendor/` 中并进行版本管理；其许可证和来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。应用代码采用 [MIT 许可证](LICENSE)；随附的课程和阅读材料仍保留各自的作者归属与使用条款。
