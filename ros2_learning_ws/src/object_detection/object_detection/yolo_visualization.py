import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import ast


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.create_subscription(Image, '', self.image_callback, 10)
        self.create_subscription(String, '', self.detections_callback, 10)
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_detections = None

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.update_display()

    def detections_callback(self, msg):
        self.latest_detections = ast.literal_eval(msg.data)
        self.update_display()

    def update_display(self):
        if self.latest_image is None or self.latest_detections is None:
            return

        image = self.latest_image.copy()
        for det in self.latest_detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            confidence = det['confidence']
            label = det['label']

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {confidence:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detections", image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
