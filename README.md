# 人体轨迹重建 · Motion Lab

独立静态诊断网站：16个连续段、15条父轨迹；GT、Phone、Watch信号与单端/双端OOF轨迹，时间轴、三维方向诊断、相关性和原论文指标。

默认序列UID 140006。可使用`?uid=18005`等链接定位其他序列。

## 运行与部署

无需npm或构建。运行`python -m http.server 8765`，访问`http://localhost:8765`。GitHub Pages从main分支根目录发布，.nojekyll关闭Jekyll处理。

export_data.py是可重复数据提取脚本，读取本地原数据生成data/*.json；运行`python export_data.py --data-root 原数据目录 --experiment-root 编号9和10实验目录`。发布文件不含个人身份、机器绝对路径或地图地理坐标。

## 来源和边界

详见[SOURCE_ALIGNMENT.md](SOURCE_ALIGNMENT.md)。代码独立实现，参考同学诊断页面的功能逻辑，未导入同学轨迹或复制其代码。

展示采样10Hz，原始信号90Hz。模型结果是已保存离线OOF、非实时推理；编号9/10只显示一个种子，不能代替完整论文验收。Watch只有去倾斜局部系，世界水平yaw未经核验；方向箭头默认关闭。逐marker位置及逐帧有效掩码未保留，不虚构展示。

论文验收阈值保持原标准。v14为姿态辅助历史轨道，编号9/10为q_nav输入，两者信息条件不同。未使用测试GT调整预测方向、缩放或选择候选。
