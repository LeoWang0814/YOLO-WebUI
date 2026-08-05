(() => {
  const preferenceKey = "yolov10-workbench.language";
  const root = document.documentElement;
  const originalText = new WeakMap();
  const originalAttributes = new WeakMap();
  const textAttributes = ["aria-label", "alt", "placeholder", "title"];

  // Terms in this table are reviewed product copy, not machine translation.
  // Technical field names retain their upstream spelling in parentheses.
  const zh = {
    "Train": "训练",
    "Predict": "预测",
    "Runs": "运行记录",
    "Docs": "文档",
    "Configuration": "配置",
    "Dataset": "数据集（Dataset）",
    "Model": "模型（Model）",
    "Source": "输入来源（Source）",
    "Pretrained": "预训练",
    "Pretrained model": "预训练模型（Pretrained model）",
    "Local .pt": "本地 .pt",
    "Local model path": "本地模型路径（Local model path）",
    "Choose a .pt file": "选择 .pt 文件",
    "No file selected": "未选择文件",
    "No files selected": "未选择文件",
    "Images": "图像",
    "Video": "视频",
    "Path": "路径",
    "Source path": "输入路径（Source path）",
    "Training parameters": "训练参数",
    "Prediction parameters": "预测参数",
    "Epochs": "训练轮次（Epochs）",
    "Patience": "早停耐心值（Patience）",
    "Image size": "图像尺寸（Image size）",
    "Batch": "批次大小（Batch）",
    "Workers": "数据加载线程数（Workers）",
    "Device": "设备（Device）",
    "Confidence": "置信度（Confidence）",
    "Run name": "运行名称（Run name）",
    "optional": "可选",
    "Auto": "自动",
    "Single GPU": "单块 GPU",
    "Multiple GPUs": "多块 GPU",
    "Advanced settings": "高级设置",
    "Find a parameter": "查找参数",
    "Core": "核心",
    "Optimizer": "优化器",
    "Augmentation": "数据增强",
    "Validation / Logging": "验证 / 日志",
    "Performance": "性能",
    "Other": "其他",
    "New training run": "新建训练任务",
    "New prediction run": "新建预测任务",
    "Training run": "训练任务",
    "Prediction run": "预测任务",
    "Start training": "开始训练",
    "Run prediction": "运行预测",
    "Starting": "正在启动",
    "Ready": "就绪",
    "Available": "可用",
    "Task": "任务",
    "Queue": "队列",
    "Detect": "目标检测（Detect）",
    "Command": "命令",
    "Live preview": "实时预览",
    "Results": "结果",
    "Local runtime": "本地运行环境",
    "Runtime output": "运行输出",
    "Command log": "命令日志",
    "Following output": "正在跟随输出",
    "Output idle": "输出空闲",
    "Auto-follow paused": "已暂停自动跟随",
    "At latest output": "已显示最新输出",
    "Unable to render metrics.": "无法渲染指标。",
    "Reconnecting to dataset preparation status…": "正在重新连接数据集准备状态…",
    "Runtime output will appear here after the run starts.": "任务启动后，运行输出会显示在这里。",
    "Waiting for process output…": "正在等待进程输出…",
    "Current progress": "当前进度",
    "Open run": "打开运行记录",
    "Open runs": "打开运行记录",
    "Back to runs": "返回运行记录",
    "New training": "新建训练",
    "New prediction": "新建预测",
    "Stop": "停止",
    "Stopping": "正在停止",
    "Retry": "重试",
    "Use": "使用",
    "Name in use": "名称已被使用",
    "Not started": "未启动",
    "Run slot unavailable": "运行槽位不可用",
    "Check configuration": "检查配置",
    "Unable to start run": "无法启动任务",
    "Status unavailable": "状态不可用",
    "Run disconnected": "任务已断开",
    "Disconnected": "已断开",
    "Running": "运行中",
    "Completed": "已完成",
    "Failed": "失败",
    "Stopped": "已停止",
    "Active": "活动中",
    "Process": "进程",
    "Run": "运行",
    "Best weights": "最佳权重",
    "Last weights": "最后权重",
    "Pending": "等待生成",
    "Log": "日志",
    "Unavailable": "不可用",
    "No captured log available.": "没有可用的已捕获日志。",
    "No generated artifacts": "没有生成的产物。",
    "Waiting for output": "正在等待输出",
    "Results will appear here after the run starts.": "任务启动后，结果会显示在这里。",
    "Complete the required fields to preview the command.": "请完成必填项以预览命令。",
    "Inspect": "检查",
    "Verify": "验证",
    "Prepare": "准备",
    "Dataset folder": "数据集文件夹（Dataset folder）",
    "Inspect a local folder and prepare a strict YOLOv10 Detect dataset.": "检查本地文件夹，并准备可供 YOLOv10 Detect 严格使用的数据集。",
    "Local path; source files stay unchanged.": "本地路径；源文件不会被修改。",
    "View supported formats and conversion rules": "查看支持的格式与转换规则",
    "Supports most popular detection dataset formats": "支持大多数主流目标检测数据集格式",
    "Add a dataset folder to inspect it.": "请输入数据集文件夹后进行检查。",
    "View dataset summary": "查看数据集摘要",
    "Learn how this format is detected and converted": "了解此格式的识别与转换方式",
    "Dataset preparation is blocked": "数据集准备被阻止",
    "Read supported formats and conversion rules": "查看支持的格式与转换规则",
    "Ready for YOLOv10 Detect": "已可用于 YOLOv10 Detect",
    "Inspecting and preparing dataset…": "正在检查并准备数据集…",
    "Preparing a local YOLOv10 Detect dataset. You can continue configuring the run.": "正在准备本地 YOLOv10 Detect 数据集；你可以继续配置任务。",
    "No CUDA devices detected": "未检测到 CUDA 设备",
    "Runtime status": "运行环境状态",
    "Theme": "主题",
    "Switch theme": "切换主题",
    "Switch language": "切换语言",
    "Switch to English": "切换为 English",
    "Switch to Chinese": "切换为中文",
    "Switch to dark theme": "切换到深色主题",
    "Switch to light theme": "切换到浅色主题",
    "Preview": "预览",
    "Media": "媒体",
    "Image": "图像",
    "Fit to view": "适应视图",
    "Open original": "打开原始文件",
    "Open video viewer": "打开视频查看器",
    "Media unavailable": "媒体不可用",
    "Zoom out": "缩小",
    "Zoom in": "放大",
    "Fit image to view": "使图像适应视图",
    "Close viewer": "关闭查看器",
    "Processed detection output": "已处理的检测输出",
    "Prediction video": "预测视频",
    "New run": "新建任务",
    "Search runs": "搜索运行记录",
    "Filter runs": "筛选运行记录",
    "All": "全部",
    "Type": "类型",
    "Name": "名称",
    "Updated": "更新时间",
    "Artifacts": "产物",
    "No local runs": "没有本地运行记录",
    "No matching runs": "没有匹配的运行记录",
    "Search documentation": "搜索文档",
    "Documentation navigation": "文档导航",
    "Documentation search results": "文档搜索结果",
    "Breadcrumb": "面包屑导航",
    "On this page": "本页内容",
    "Related documentation": "相关文档",
    "Page": "页面",
    "Getting started": "快速开始",
    "Product overview": "产品概览",
    "Dataset preparation": "数据集准备",
    "Models and weights": "模型与权重",
    "Runs and results": "运行记录与结果",
    "Configuration reference": "配置参考",
    "Runtime and hardware": "运行环境与硬件",
    "Files, storage and privacy": "文件、存储与隐私",
    "Troubleshooting": "故障排查",
    "Glossary and limits": "术语与限制",
    "START HERE": "从这里开始",
    "WORKFLOWS": "工作流",
    "REFERENCE": "参考资料",
    "ON THIS PAGE": "本页内容",
    "Requirements": "环境要求",
    "Launch the Workbench": "启动 Workbench",
    "First training run": "首次训练任务",
    "First prediction": "首次预测",
    "What it is": "产品定位",
    "Workflow": "工作流",
    "Product limits": "产品限制",
    "Before starting": "开始前准备",
    "Configuration form": "配置表单",
    "Execution and progress": "执行与进度",
    "Training outputs": "训练输出",
    "From zero to Ready": "从零到就绪",
    "Preparation progress": "准备进度",
    "Supported formats": "支持的格式",
    "Strict lossless policy": "严格无损策略",
    "Cache and output": "缓存与输出",
    "Common blockers": "常见阻塞项",
    "Pretrained models": "预训练模型",
    "Local models and uploads": "本地模型与上传",
    "Downloads and verification": "下载与校验",
    "Model errors": "模型错误",
    "Choose a source": "选择输入来源",
    "Prediction settings": "预测设置",
    "Run and inspect": "运行与查看",
    "Media viewer": "媒体查看器",
    "Run list": "运行列表",
    "Run details": "运行详情",
    "Run states": "运行状态",
    "Primary controls": "主要控件",
    "Train advanced settings": "训练高级设置",
    "Predict advanced settings": "预测高级设置",
    "Workbench-managed fields": "Workbench 托管字段",
    "Device selection": "设备选择",
    "Performance controls": "性能控制",
    "Run queue": "运行队列",
    "Model downloads": "模型下载",
    "Storage locations": "存储位置",
    "Data lifecycle": "数据生命周期",
    "Privacy and access": "隐私与访问",
    "Retention": "保留策略",
    "Dataset errors": "数据集错误",
    "Run errors": "运行错误",
    "Output errors": "输出错误",
    "Glossary": "术语表",
    "Supported capabilities": "支持的能力",
    "Not supported": "暂不支持",
    "Scope": "适用范围",
    "One active run": "一次只能运行一个任务",
    "Ready means trainable": "就绪表示可以训练",
    "Stopping": "停止任务",
    "Recognized can still mean blocked": "已识别并不代表可以转换",
    "No URL source mode": "不支持 URL 输入来源",
    "Blocked or ambiguous": "被阻止或格式存在歧义",
    "Missing or duplicate image": "图像缺失或重复",
    "Invalid box": "边界框无效",
    "Empty train/validation split": "训练 / 验证划分为空",
    "Unexpected cached result": "缓存结果不符合预期",
    "Artifact": "产物（Artifact）",
    "Cache": "缓存（Cache）",
    "Prepared dataset": "准备好的数据集（Prepared dataset）",
    "Split": "数据划分（Split）",
    "Format": "格式",
    "Family": "格式家族",
    "Status": "状态",
    "Strict rule": "严格规则",
    "Release": "版本",
    "Filename": "文件名",
    "Size": "大小",
    "Location": "位置",
    "Purpose": "用途",
    "Mode": "模式",
    "Control": "控件",
    "Group": "分组",
    "Field": "字段",
    "Default": "默认值",
    "Required": "必填",
    "Type": "类型",
    "Get started": "快速开始",
    "A local workflow for YOLO Detect": "面向 YOLO Detect 的本地工作流",
    "Create, monitor, stop, and review an axis-aligned detection training run.": "创建、监控、停止并查看轴对齐目标检测训练任务。",
    "Inspect one local folder, prepare a strict YOLO Detect dataset, and keep the source unchanged.": "检查一个本地文件夹，准备严格的 YOLO Detect 数据集，并保持源文件不变。",
    "Choose verified pretrained weights or a local checkpoint without losing track of where files are stored.": "选择已验证的预训练权重或本地检查点，同时清楚了解文件的存储位置。",
    "Run a chosen model against local images, video, or a filesystem path and inspect saved outputs.": "对本地图像、视频或文件系统路径运行所选模型，并查看已保存的输出。",
    "Find completed work, read its command and logs, and inspect metrics or generated media.": "查找已完成任务，查看命令和日志，并检查指标或生成的媒体。",
    "Primary controls and every Advanced settings field currently exposed by the Workbench.": "Workbench 当前提供的主要控件及全部高级设置字段。",
    "How local execution, device selection, downloads, and the single-run queue behave.": "本地执行、设备选择、下载以及单任务队列的运行方式。",
    "Where the Workbench stores data and what stays on your machine.": "Workbench 的数据存储位置，以及哪些数据保留在本机。",
    "Resolve common dataset, model, source, runtime, and output failures with the evidence shown in the UI.": "结合界面中显示的信息，解决常见的数据集、模型、输入来源、运行环境和输出问题。",
    "Shared vocabulary and the current product boundaries for this local YOLO Detect Workbench.": "此本地 YOLO Detect Workbench 的共享术语与当前产品边界。",
    "Use Python 3.10 or later with the dependencies in": "请使用 Python 3.10 或更高版本，并安装",
    ". The Workbench is self-hosted: it has no sign-in, no cloud project store, and no multi-user coordination layer.": "中的依赖。Workbench 为自托管应用：没有登录、云端项目存储或多人协作层。",
    "Use this application for local, axis-aligned YOLO Detect training and prediction. It does not add segmentation, pose, classification, OBB, or hosted collaboration workflows.": "本应用用于本地轴对齐 YOLO Detect 训练与预测；不提供分割、姿态、分类、OBB 或托管协作工作流。",
    "From the project directory, start the application with:": "在项目目录中，使用以下命令启动应用：",
    "The default address is": "默认地址为",
    ". Set": "。请在启动前设置",
    "and": "和",
    "before launch to override the bind address. A direct Uvicorn launch is also supported for a custom local port.": "以覆盖绑定地址；也支持通过 Uvicorn 直接启动并指定本地端口。",
    "Your first training run": "首次训练任务",
    "Open": "打开",
    "and enter one local dataset folder.": "，然后输入一个本地数据集文件夹。",
    "Select": "选择",
    ". Continue only when the dataset result is": "。仅当数据集结果显示为",
    ".": "。",
    "Select a pretrained model or local": "选择预训练模型或本地",
    "checkpoint, then choose epochs, image size, batch, workers, and device.": "检查点，然后设置训练轮次、图像尺寸、批次大小、数据加载线程数和设备。",
    "Review the live command preview, optionally enter a run name, and select": "查看实时命令预览，可选填运行名称，然后选择",
    "Follow download, epoch, and log progress. Open the completed run from": "跟踪下载、训练轮次和日志进度；完成后可从",
    "First prediction": "首次预测",
    "and select a model.": "，然后选择模型。",
    "Provide uploaded images, one video, or a local source path.": "提供上传的图像、一个视频或本地输入路径。",
    "Set confidence, IoU, image size, and device as needed.": "按需设置置信度、IoU、图像尺寸和设备。",
    "then inspect generated media in the Results panel or Runs archive.": "，然后在“结果”面板或运行记录中查看生成的媒体。",
    "The Workbench manages one Ultralytics training or prediction process at a time. Wait for it to finish or stop it before starting another run.": "Workbench 同一时间只管理一个 Ultralytics 训练或预测进程；请等待其完成或停止后再启动另一个任务。",
    "The Workbench is a local interface around the bundled Ultralytics CLI. It prepares compatible detection data, resolves a model, creates a timestamped run directory, and displays live process output and artifacts.": "Workbench 是围绕内置 Ultralytics CLI 的本地界面：它准备兼容的检测数据、解析模型、创建带时间戳的运行目录，并显示实时进程输出和产物。",
    "Prepares a local detection dataset and runs": "准备本地检测数据集并运行",
    "Runs": "运行记录",
    "against local media or a path.": "以处理本地媒体或路径。",
    "Reads persisted artifacts under the local run archive.": "读取本地运行归档中的持久化产物。",
    "Workflow lifecycle": "工作流生命周期",
    "Configure a form and inspect its generated command.": "配置表单并检查生成的命令。",
    "Start one managed run; the Workbench creates an isolated local run directory.": "启动一个受管任务；Workbench 会创建独立的本地运行目录。",
    "Resolve or download the selected model, then launch the CLI process.": "解析或下载所选模型，然后启动 CLI 进程。",
    "Persist command metadata and log output while the process runs.": "在进程运行期间保存命令元数据和日志输出。",
    "Review metrics, weights, images, video, and logs after completion.": "完成后查看指标、权重、图像、视频和日志。",
    "Only axis-aligned object detection is trainable through this interface.": "此界面仅支持训练轴对齐目标检测。",
    "Prediction supports local uploads and paths, not URL sources.": "预测支持本地上传和路径，不支持 URL 来源。",
    "Dataset preparation is local-only and will block ambiguous or lossy conversions.": "数据集准备仅在本地进行，并会阻止有歧义或有损的转换。",
    "Stopping a run requests process termination; partial artifacts may remain in that run directory.": "停止任务会请求终止进程；部分产物可能仍保留在该运行目录中。",
    "A training run requires a Ready dataset result and a valid model selection. The Start button validates the dataset again before launch, so a changed source folder cannot silently reuse stale configuration.": "训练任务需要数据集处于“就绪”状态并选择有效模型。启动前会再次验证数据集，因此源文件夹变更后不会悄然复用过期配置。",
    "Ready confirms an axis-aligned detection dataset, valid coordinates, class mapping, and a managed generated data YAML.": "“就绪”表示已确认轴对齐检测数据集、坐标、类别映射均有效，并已生成受管理的 data YAML。",
    "Enter one folder, select Inspect, and use its managed prepared output.": "输入一个文件夹，选择“检查”，并使用其受管理的准备输出。",
    "Choose verified pretrained weights, an existing local path, or upload a": "选择已验证的预训练权重、已有本地路径，或上传一个",
    "file.": "文件。",
    "Epochs set the maximum duration; patience controls early stopping; image size, batch, workers, and device affect throughput and memory.": "训练轮次决定最长时长；早停耐心值控制提前停止；图像尺寸、批次大小、数据加载线程数和设备会影响吞吐量与内存。",
    "Use only when a primary control is insufficient. Every exposed field is listed in the": "仅当主要控件不足时使用。所有已暴露字段均列在",
    ".": "。",
    "The live preview reflects form values before the run starts; it is informational and does not execute a shell command in the browser.": "实时预览反映任务启动前的表单值；它仅供参考，不会在浏览器中执行 shell 命令。",
    "After start, the inspector shows the run name, process state, best/last weights when available, and a Stop action. The progress card can report model download, epoch progress, or an indeterminate phase. The terminal follows new output until you scroll away.": "启动后，检查器会显示运行名称、进程状态、可用时的最佳/最后权重以及“停止”操作。进度卡会显示模型下载、训练轮次进度或不确定阶段；终端会持续跟随新输出，直到你滚动离开。",
    "Select Stop to request termination. The current run becomes stopped after the process exits; inspect its log and partial outputs in Runs.": "选择“停止”以请求终止。进程退出后，当前任务将变为“已停止”；可在运行记录中检查其日志和部分输出。",
    "Completed training runs can include": "已完成的训练任务可能包含",
    ", generated plots, command metadata, and logs. When": "、生成图表、命令元数据和日志。当",
    "is available, the Workbench renders loss, quality, and learning-rate charts.": "可用时，Workbench 会渲染损失、质量和学习率图表。",
    "Place local images and their annotation export in one folder.": "将本地图像及其标注导出文件放入同一文件夹。",
    "Enter that folder in Train and select": "在“训练”中输入该文件夹并选择",
    "The Workbench identifies one format from the folder layout and annotation schema.": "Workbench 会根据文件夹布局和标注模式识别一种格式。",
    "It validates images, classes, boxes, and split leakage, then creates a prepared cache when valid.": "它会验证图像、类别、边界框和数据划分泄漏；验证通过后创建准备缓存。",
    "Use the Ready result to start training; source files remain unchanged.": "使用“就绪”结果启动训练；源文件保持不变。",
    "The progress card reports the active work, including folder checks, format identification, source-file verification, record validation, and prepared-file creation. A completed cache can be reused when its source path, file names, sizes, and modification timestamps are unchanged.": "进度卡会报告当前工作，包括文件夹检查、格式识别、源文件验证、记录验证和准备文件创建。当源路径、文件名、大小和修改时间未变时，可以复用已完成的缓存。",
    "Only trainable axis-aligned detection data is converted. The Workbench preserves image bytes, declared classes, valid declared splits, and box geometry. It never guesses coordinate order, clamps out-of-bounds boxes, downloads remote images, or turns oriented boxes into enclosing rectangles.": "仅转换可训练的轴对齐检测数据。Workbench 会保留图像字节、声明的类别、有效的数据划分和边界框几何；它不会猜测坐标顺序、截断越界框、下载远程图像或将旋转框变成外接矩形。",
    "OBB, multi-label classification, segmentation-only, unknown VLM grammar, missing media, malformed boxes, and ambiguous schemas are reported as incompatible instead of being approximated.": "OBB、多标签分类、仅分割数据、未知 VLM 语法、媒体缺失、边界框畸形和模式歧义都会被报告为不兼容，而不会被近似处理。",
    "Use one annotation format per folder; separate mixed exports.": "每个文件夹只使用一种标注格式；请分离混合导出文件。",
    "Ensure every referenced image can be located uniquely.": "确保每个被引用图像都可以被唯一定位。",
    "Fix empty or degenerate boxes instead of relying on automatic repair.": "请修复空边界框或退化边界框，不要依赖自动修复。",
    "Declare meaningful train and validation splits, or provide at least two images for stable generated splits.": "声明有效的训练与验证划分，或至少提供两张图像以稳定生成划分。",
    "See": "请参阅",
    "for recovery steps.": "以了解恢复步骤。",
    "Choose a model from the catalog below. The form reports whether the expected weight file is already cached; a missing model downloads only when its run starts.": "从下方目录中选择模型。表单会显示预期权重文件是否已缓存；缺失模型只会在任务启动时下载。",
    "Pretrained downloads use the catalog source, retry transient network failures, resume compatible partial downloads, and verify SHA-256 before the model becomes cached. The run progress card shows download percentage or transfer speed, followed by checksum verification.": "预训练下载使用目录来源、会重试瞬时网络失败、恢复兼容的部分下载，并在模型进入缓存前校验 SHA-256。任务进度卡会显示下载百分比或传输速度，随后进行校验和验证。",
    "Remote URLs are intentionally unsupported. Use a local downloaded file or folder instead.": "产品有意不支持远程 URL；请改用本地下载的文件或文件夹。",
    "Confidence controls the minimum retained detection score. IoU controls non-maximum suppression. Image size and device affect latency and memory. Use Advanced settings for output labels, crops, class filtering, video stride, augmented inference, or agnostic NMS.": "置信度控制保留检测结果的最低分数；IoU 控制非极大值抑制。图像尺寸和设备会影响延迟与内存。可在高级设置中调整输出标签、裁剪、类别筛选、视频步长、增强推理或类别无关 NMS。",
    "Select an output image to open the viewer. Use zoom controls or the mouse wheel to zoom, drag a zoomed image to pan, reset to fit, or open the original artifact in a separate browser view. Generated video is converted to a browser-friendly MP4 when possible; the original remains available if conversion is unavailable.": "选择输出图像以打开查看器。可使用缩放控件或鼠标滚轮缩放，拖动已缩放图像进行平移，重置为适应视图，或在单独浏览器视图中打开原始产物。生成的视频会在可能时转换为浏览器友好的 MP4；若无法转换，原文件仍可使用。",
    "Lets Ultralytics choose the available execution device.": "让 Ultralytics 选择可用执行设备。",
    "Forces CPU inference or training; expect slower training and inference.": "强制使用 CPU 推理或训练；训练和推理速度会较慢。",
    "Uses one detected CUDA device by numeric index.": "按数字索引使用一块已检测到的 CUDA 设备。",
    "Passes the selected CUDA device indices to the CLI. This option is only meaningful when CUDA devices are detected.": "将所选 CUDA 设备索引传给 CLI；仅在检测到 CUDA 设备时有意义。",
    "The application permits one active local Ultralytics process. This prevents a training and prediction process from competing for the same GPU, model files, logs, and output resources. A new request receives a visible conflict until the active run reaches a terminal state.": "应用只允许一个活动的本地 Ultralytics 进程。这可避免训练和预测进程争用同一 GPU、模型文件、日志和输出资源。在活动任务进入终止状态前，新请求会显示明确的冲突。",
    "Dataset inspection, run execution, media processing, and artifact viewing happen on the machine hosting the Workbench. The app does not upload dataset images or fetch remote prediction URLs. Pretrained model downloads are the only normal network operation and use the configured catalog source. The application has no built-in authentication layer, so bind it only to an address appropriate for your environment.": "数据集检查、任务执行、媒体处理和产物查看均在运行 Workbench 的机器上进行。应用不会上传数据集图像或请求远程预测 URL。预训练模型下载是唯一的常规网络操作，并使用配置的目录来源。应用没有内置认证层，因此只应绑定到适合你环境的地址。",
    "Prediction URL input or automatic remote-media fetching.": "预测 URL 输入或自动获取远程媒体。",
    "Lossy conversion from OBB, segmentation-only, pose, or image-level classification data to Detect labels.": "将 OBB、仅分割、姿态或图像级分类数据有损转换为 Detect 标签。",
    "Training non-Detect tasks through this Workbench.": "通过此 Workbench 训练非 Detect 任务。",
    "Concurrent managed training and prediction processes.": "并发的受管训练和预测进程。",
    "Built-in accounts, cloud synchronization, or collaboration controls.": "内置账户、云同步或协作控制。",
  };

  const patterns = [
    [/^(\d+) images$/, "$1 张图像"],
    [/^(\d+) objects$/, "$1 个目标"],
    [/^(\d+) classes$/, "$1 个类别"],
    [/^(\d+) GPU$/, "$1 块 GPU"],
    [/^Saved (.+)$/, "已保存 $1"],
    [/^Exit (-?\d+)$/, "退出码 $1"],
    [/^Model download (\d+)%$/, "模型下载 $1%"],
  ];

  const isChinese = () => root.dataset.language === "zh";
  const translate = (value) => {
    if (!isChinese() || typeof value !== "string") return value;
    if (zh[value]) return zh[value];
    for (const [pattern, replacement] of patterns) {
      if (pattern.test(value)) return value.replace(pattern, replacement);
    }
    return value;
  };

  const shouldSkip = (node) => node.parentElement?.closest("script, style, code, pre, [data-i18n-skip]");

  const translateText = (scope) => {
    const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (shouldSkip(node)) return;
      const original = originalText.get(node) ?? node.nodeValue;
      originalText.set(node, original);
      const value = original.trim();
      if (!value) return;
      node.nodeValue = original.replace(value, translate(value));
    });
  };

  const translateAttributes = (scope) => {
    const descendants = scope.querySelectorAll ? [...scope.querySelectorAll("*")] : [];
    const elements = [scope, ...descendants];
    elements.forEach((element) => {
      if (!(element instanceof Element) || element.closest("[data-i18n-skip]")) return;
      const saved = originalAttributes.get(element) || {};
      textAttributes.forEach((attribute) => {
        if (!element.hasAttribute(attribute)) return;
        if (!(attribute in saved)) saved[attribute] = element.getAttribute(attribute);
        const original = saved[attribute];
        element.setAttribute(attribute, translate(original));
      });
      originalAttributes.set(element, saved);
    });
  };

  const updateToggle = () => {
    const toggle = document.querySelector("[data-language-toggle]");
    if (!toggle) return;
    const label = isChinese() ? "Switch to English" : "Switch to Chinese";
    toggle.setAttribute("aria-label", translate(label));
    toggle.setAttribute("title", translate(label));
    toggle.querySelector("[data-language-zh]")?.classList.toggle("is-active", isChinese());
    toggle.querySelector("[data-language-en]")?.classList.toggle("is-active", !isChinese());
  };

  const apply = (scope = document) => {
    root.lang = isChinese() ? "zh-CN" : "en";
    translateText(scope);
    translateAttributes(scope);
    updateToggle();
  };

  const setLanguage = (language) => {
    root.dataset.language = language;
    localStorage.setItem(preferenceKey, language);
    apply(document);
    document.dispatchEvent(new CustomEvent("workbench:languagechange", { detail: { language } }));
  };

  const initialize = () => {
    apply(document);
    document.querySelectorAll("[data-language-toggle]").forEach((toggle) => {
      if (toggle.dataset.bound) return;
      toggle.dataset.bound = "true";
      toggle.addEventListener("click", () => setLanguage(isChinese() ? "en" : "zh"));
    });
  };

  window.WorkbenchI18n = { apply, initialize, isChinese, setLanguage, t: translate };
})();
