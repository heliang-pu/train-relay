# train-relay

[English](README.md) | **中文**

在一台机器上写代码、在另一台机器上训练时用的伴随工具：通过 ssh 在远端起一个长任务，盯着它，把结果拉回本机。
一个任务写成一份 TOML 文件，剩下的一条命令完成：

```bash
./train-relay run job.toml
```

它取代的是每次训练都要重写一遍的那份 shell 脚本：等显卡空出来、用 `nohup` 起训、隔几分钟 `tail` 一下日志、
`scp` 权重、再确认文件传完整了。

只用 Python 标准库（3.11 以上），不用安装。远端只需要 bash、dd、stat、md5sum；用到显卡检查时还要 nvidia-smi。

## 六个阶段

1. **起训前检查。** 远端必须存在的文件（可以带 sha256）、必须不存在的路径。这样不会用错起始权重，也不会覆盖已有的训练。
2. **显卡。** 需要的空闲显卡数量，隔 20 秒看两次。不够时报错退出，或者按配置一直等到空出来。
3. **启动。** 命令在远端用 `setsid nohup` 脱离会话运行，留下进程号文件和日志。合上笔记本不影响它。
4. **盯进度。** 每次轮询看四件事：是否结束、日志里有没有报错行、进程是否还在、进度到哪了。每次打印一行。
5. **拉结果。** 结果文件出现后以及结束时拉回本机，只拉最新的 N 个。每个文件分成多段，经多条 ssh 连接并行传，传完和远端核对 md5。
6. **钩子。** 每拉完一个文件、任务成功结束、任务失败时，各执行一条本机命令：发提醒、登记权重、启动下一步。

## 用法

```bash
git clone https://github.com/heliang-pu/train-relay && cd train-relay

./train-relay run job.toml              # 全流程；建议放在 tmux 里或用 nohup 运行
./train-relay run job.toml --dry-run    # 只打印计划和远端启动脚本，什么都不碰
./train-relay run job.toml --attach     # 任务已经在跑：只盯进度和拉结果
./train-relay status job.toml           # 看一眼现在的状态
./train-relay pull job.toml             # 现在就把最新的结果拉回来
./train-relay stop job.toml             # 停掉远端任务（整个进程组）
```

本机这边的 `run` 被打断（Ctrl-C、终端关了、VPN 断了）不会影响远端的任务，之后用 `--attach` 接着盯。
查状态时 ssh 连不上不算任务失败，下一次轮询再试。

完整的配置例子见 [`examples/finetune.toml`](examples/finetune.toml)。

## 配置项

| 段 | 键 | 含义 | 默认 |
|---|---|---|---|
| 顶层 | `name` | 任务名，只能用字母、数字、点、横线、下划线 | 必填 |
| 顶层 | `host` | ssh 别名；写 `local` 则在本机执行 | 必填 |
| 顶层 | `workdir` | 远端工作目录（绝对路径），命令在这里执行 | 必填 |
| `[preflight]` | `exists` | 必须存在的远端路径 | 无 |
| `[preflight]` | `absent` | 必须不存在的远端路径 | 无 |
| `[preflight.sha256]` | `"路径" = "哈希"` | 文件必须是这个 sha256 | 无 |
| `[gpus]` | `count` | 需要的空闲显卡数；0 表示不检查 | 0 |
| `[gpus]` | `idle_mib` | 显存占用低于这个值算空闲 | 2000 |
| `[gpus]` | `wait` | 不够时一直等，而不是报错 | false |
| `[launch]` | `command` | 要执行的命令（可以多行） | 必填 |
| `[launch.env]` | | 导出给命令的环境变量 | 无 |
| `[watch]` | `poll_s` | 多久看一次，单位秒 | 300 |
| `[watch]` | `done` | 日志里表示成功结束的正则；不写则以退出码 0 为准 | 无 |
| `[watch]` | `error` | 日志里表示出错的正则 | 无 |
| `[watch]` | `progress` | 取进度用的正则，打印最后一次匹配 | 无 |
| `[watch]` | `settle_s` | 运行中拉文件时，文件多久没变才算写完 | 60 |
| `[[pull]]` | `glob` | 远端结果文件的通配路径（绝对路径） | 必填 |
| `[[pull]]` | `to` | 本机目录 | 必填 |
| `[[pull]]` | `newest` | 只拉修改时间最新的 N 个；0 表示全部 | 2 |
| `[[pull]]` | `while_running` | 训练过程中就开始拉 | false |
| `[[sync]]` | `from`、`to` | 用 rsync 同步的小文件或目录 | 无 |
| `[transfer]` | `parallel` | 每个文件分几段并行传 | 8 |
| `[transfer]` | `min_free_gb` | 拉完后本机剩余空间会低于这个值就不拉 | 0 |
| `[hooks]` | `after_pull`、`on_done`、`on_fail` | 本机命令 | 无 |

钩子命令都能拿到 `RELAY_NAME` 和 `RELAY_HOST`；`after_pull` 另有 `RELAY_FILE`（本机路径）和 `RELAY_REMOTE`；
`on_done` 和 `on_fail` 另有 `RELAY_STATE` 和 `RELAY_TEXT`（一句话的结果）。

## 怎么判断任务的状态

| 判定 | 条件 |
|---|---|
| 失败 | 日志里出现 `error` 匹配的行 |
| 成功 | 日志里出现 `done` 匹配的行；没配 `done` 时，命令以退出码 0 结束 |
| 失败 | 命令以非 0 退出；或者配了 `done`，命令以 0 退出但那一行没出现 |
| 失败 | 进程不在了，也没有留下退出记录（被杀掉，或者机器重启了） |
| 运行中 | 以上都不是 |

## 远端和本机各留下什么

- 远端：`<workdir>/.train-relay/<name>.sh`（启动脚本）、`<name>.pid`、`<name>.log`（命令的全部输出）。
- 本机：每个 `to` 目录里的 `.train-relay.json` 记录每个已拉文件的大小、md5、来源和时间。已经拉过且远端没变的文件不会重拉。
- 传输中的文件叫 `.<文件名>.part`，md5 对上之后才改成正式的名字，对不上就删掉。坏掉的下载不会被当成权重。

## 注意

- `while_running = true` 配合 `newest = 2` 时，训练中每出现一个新文件都会被拉，因为它在那一刻属于最新的两个。
  轮数多、文件大、流量计费时，把 `while_running` 关掉。
- `stop` 结束的是启动脚本所在的进程组。训练框架自己另开会话的子进程不在其中。
- 这个工具只管一个任务。多台机器上的任务排队见 [gpuq](https://github.com/heliang-pu/gpuq)。

## 测试

```bash
python3 -m pytest -q
```

测试全部在本机进行（`host = "local"`），用一条假的训练命令写日志和结果文件，覆盖成功、各种失败、等显卡、接管、
停止和传输损坏这些情况。
