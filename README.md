# Kaguya

Kaguya 是一套可复现部署的 QQ 群聊机器人：使用 NapCat 接收 OneBot 11 事件，独立适配器负责准入、会话上下文、历史查询、主动参与判断与消息投递，Hermes Agent 负责推理、人格和记忆。

仓库来自一套实际运行中的机器人，但它是经过脱敏的公开发行版。**不含**真实 QQ 号或群号、现有群绑定、聊天记录、上下文图片、登录态、API Key、访问令牌、数据库、日志，以及针对真实群成员的关系或偏好记忆。

## 包含什么

- `onebot_adapter/`：NapCat ↔ Hermes 的完整 OneBot 运行时、消息查询函数、唤醒/跳过判断、上下文补全、群聊队列和 WebUI。
- `profiles/aidog/`、`profiles/kig/`：两套群聊人格、规则和安全的空白记忆种子。
- `profiles/qq-observer/`：私有的群聊参与判断资料。
- `profiles/default/`：通用根资料，作为公开部署的默认基线。
- `overlays/hermes/`：当前服务器曾使用的 Hermes 官方 QQ Bot 平台增量文件；NapCat/OneBot 主线路不依赖它。
- `dependencies.lock.json`、`requirements.lock`、`frontend/package-lock.json`：Hermes、NapCat、Python 与前端依赖的固定版本。
- `kaguya_toolkit/`：安装、初始化、群绑定、NapCat 配置、启停、诊断和隐私扫描命令。

Hermes 自带技能由锁定版本的 Hermes 安装器提供，仓库不重复复制。Kaguya 自己需要的 OneBot 工具随适配器插件安装。

## 环境

主要支持 Ubuntu 24.04 / Debian 系 Linux，Python 3.11 及以上。建议至少准备：

- `git`、`python3`、`python3-venv`、`curl`
- Node.js/npm（用于构建 WebUI；没有时机器人核心仍可安装）
- 桌面 QQ/NapCat 所需的系统依赖；无图形界面服务器还需要 `xvfb`

NapCat 和非官方 QQ 自动化可能受 QQ 平台规则、客户端版本与账号风控影响。请只在自己有权管理的账号和群中使用。

## 安装

```bash
git clone https://github.com/hoshyo/Kaguya.git
cd Kaguya
chmod +x ./kaguya
./kaguya install --with-napcat
```

`install` 会：

1. 以 SHA-256 校验后安装锁定提交的 Hermes；
2. 可选运行锁定提交的 NapCat 安装器；
3. 配置根 `default` 资料，并创建 `aidog`、`kig`、`qq-observer` 三个隔离的 Hermes profile；
4. 用软链接接入四份人格/判断资料、profile 定义和 OneBot 工具插件；
5. 把可变记忆种子复制到各 profile，而不是反向写回 Git；
6. 在 `~/.config/kaguya/` 生成仅属于本机的令牌和适配器配置；
7. 生成 systemd 用户服务。

如果 Hermes 尚未配置模型，初始化时会进入 Hermes 自己的设置向导。也可以随后单独执行：

```bash
./kaguya auth
```

仓库不会读取或迁移 Codex/ChatGPT 登录，也不会自动采用其他程序的登录状态。

## 绑定资料与群

真实群号只写入本机的 `~/.config/kaguya/onebot.json` 和 `~/.hermes/config.yaml`，不会写入仓库。

```bash
./kaguya bind --group 你的QQ群号 --profile aidog --name 本地备注名
./kaguya bind --group 另一个QQ群号 --profile kig --name 本地备注名
```

解除绑定：

```bash
./kaguya unbind --group 你的QQ群号
```

`aidog` 与 `kig` 是两套预设，不绑定公开仓库作者原来的任何群。记忆也按 profile 隔离。

## 连接 NapCat

先在 NapCat 中登录你自己的 QQ，再让 Kaguya 为它添加一条本地反向 WebSocket。可以传单个 `onebot11*.json`，也可以传包含这些文件的目录：

```bash
./kaguya napcat --config ~/Napcat/config
```

命令会先把原文件备份到 `~/.config/kaguya/backups/`，然后写入新生成的本机令牌。不会把 QQ 登录态复制进仓库。

## 启动与检查

```bash
./kaguya on
./kaguya status
./kaguya doctor
```

停止：

```bash
./kaguya off
```

如果你的 NapCat 可执行文件不在默认位置，可重新初始化并明确指定：

```bash
./kaguya init --napcat-executable /path/to/qq
```

已有同名 Hermes profile 时，初始化默认拒绝覆盖。确认要替换由本仓库管理的链接/文件时才使用 `--force-existing`；被替换内容会先保存到本机备份目录。

## 数据边界

运行后，本机会自然产生敏感数据：群消息归档、成员标识、profile 记忆、令牌、日志和 QQ 登录态。它们均位于仓库之外或被 `.gitignore` 排除。发布前运行：

```bash
./kaguya privacy-check
```

隐私扫描会拒绝常见密钥、真实样式的 profile QQ/群标识、数据库、日志、压缩包和媒体文件。更完整的边界见 [PRIVACY.md](PRIVACY.md) 与 [SECURITY.md](SECURITY.md)。

## 开发

```bash
./kaguya bootstrap
.venv/bin/ruff check .
.venv/bin/pytest -q
cd frontend && npm ci && npm run build
```

适配器详细配置和 API 见 [docs/README.md](docs/README.md)。Python 代码采用 MIT 许可证；依赖项目及 QQ/NapCat/Hermes 各自遵循其许可证和服务条款，详见 [NOTICE.md](NOTICE.md)。
