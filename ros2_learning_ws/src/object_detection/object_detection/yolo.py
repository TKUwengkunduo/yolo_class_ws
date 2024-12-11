import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO


class YOLODetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(String, '', 10)
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")  # 加載 YOLO 模型
        self.get_logger().info("YOLO Detection Node has started.")

    def image_callback(self, msg):
        self.get_logger().info("Received image data.")
        # 將 ROS 影像訊息轉為 OpenCV 格式
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 使用 YOLO 進行預測
        results = self.model(cv_image)
        detections = results[0].boxes.data.cpu().numpy()

        # 整理檢測結果
        output = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"{self.model.names[int(cls)]}"
            output.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(conf),
                "label": label
            })

        # 發布檢測結果
        self.publisher.publish(String(data=str(output)))
        self.get_logger().info(f"Published detections: {output}")


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
