import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import ast  # 用於安全地解析字串為 Python 資料結構


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.image_subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10
        )
        self.detection_subscription = self.create_subscription(
            String,
            '/detections',
            self.detections_callback,
            10
        )
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_detections = None
        self.get_logger().info("Visualization Node has started.")

    def image_callback(self, msg):
        self.get_logger().info("Received image.")
        # 將 ROS 影像消息轉為 OpenCV 格式
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.update_display()

    def detections_callback(self, msg):
        self.get_logger().info("Received detections.")
        # 解析字串為 Python 資料結構
        try:
            self.latest_detections = ast.literal_eval(msg.data)
        except (ValueError, SyntaxError) as e:
            self.get_logger().error(f"Failed to parse detections: {e}")
            self.latest_detections = None

        self.update_display()

    def update_display(self):
        # 確保影像和檢測結果都已接收
        if self.latest_image is None or self.latest_detections is None:
            return

        try:
            image = self.latest_image.copy()
            detections = self.latest_detections

            # 繪製每個檢測框
            for det in detections:
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                confidence = det['confidence']
                label = det['label']

                # 繪製邊框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 繪製標籤和置信度
                text = f"{label} {confidence:.2f}"
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 顯示影像
            cv2.imshow("Detections", image)
            cv2.waitKey(1)
        except KeyError as e:
            self.get_logger().error(f"Error in drawing detections: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
