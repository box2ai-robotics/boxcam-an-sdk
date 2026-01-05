# BoxCam_AN SDK 示例代码

## basic_usage.py

基础使用示例，展示如何获取三个图像：
- RGB灰度图
- RGB彩色图  
- 近场感知图

## robot_wrist_camera.py

机器人腕部相机封装示例，包含：
- `RobotWristCamera` 类：封装相机操作
- `get_images()`: 获取三个图像
- `get_near_field_distance()`: 获取原始距离数据

## 运行示例

```bash
# 基础示例
python example/basic_usage.py

# 机器人腕部相机示例
python example/robot_wrist_camera.py
```

