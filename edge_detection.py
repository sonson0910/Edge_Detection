import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class EdgeDetection:
    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image

    def compute_gradient(self, image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return gradient_magnitude

    def build_cost_function(self, gradient):
        cost = -gradient 
        return cost

    def dynamic_programming_edge_detection(self, cost):
        rows, cols = cost.shape
        dp = np.full_like(cost, float('inf'))  # Ma trận lưu chi phí tối thiểu
        path = np.zeros((rows, cols, 2), dtype=int)  # Ma trận lưu hướng đi: 2 chiều để lưu tọa độ (di, dj)
        visited = np.zeros_like(cost, dtype=bool)  # Ma trận lưu các điểm đã đi qua

        # Hướng di chuyển: lên, xuống, trái, phải, chéo trên-trái, chéo trên-phải, chéo dưới-trái, chéo dưới-phải
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Khởi tạo hàng đợi với các điểm của hàng đầu tiên, giữ nguyên chi phí của hàng đầu tiên
        queue = deque([(0, j) for j in range(cols)])
        dp[0, :] = cost[0, :]

        # Sử dụng hàng đợi để duyệt qua toàn bộ thực thể
        while queue:
            i, j = queue.popleft()

            for di, dj in directions:
                new_i, new_j = i + di, j + dj

                # Kiểm tra điểm có nằm trong ma trận và chưa được duyệt qua
                if 0 <= new_i < rows and 0 <= new_j < cols and not visited[new_i, new_j]:
                    # Tính chi phí mới nếu đi đến điểm (new_i, new_j)
                    new_cost = dp[i, j] + cost[new_i, new_j]
                    
                    # Cập nhật nếu tìm thấy chi phí nhỏ hơn cho điểm (new_i, new_j)
                    if new_cost < dp[new_i, new_j]:
                        dp[new_i, new_j] = new_cost
                        path[new_i, new_j] = (di, dj)  # Lưu hướng di chuyển
                        queue.append((new_i, new_j))  # Đưa điểm vào hàng đợi

            # Đánh dấu điểm (i, j) đã được duyệt
            visited[i, j] = True

        # Truy ngược lại đường đi tối ưu để bao trọn thực thể
        optimal_path = []
        min_index = np.argmin(dp[-1, :])  # Tìm vị trí có chi phí nhỏ nhất ở hàng cuối cùng
        current_row, current_col = rows - 1, min_index
        optimal_path.append((current_row, current_col))
        visited[current_row, current_col] = True

        # Truy ngược lại cho đến khi quay về điểm bắt đầu
        while (current_row, current_col) != (0, min_index):
            di, dj = path[current_row, current_col]
            current_row, current_col = current_row - di, current_col - dj
            optimal_path.append((current_row, current_col))
            visited[current_row, current_col] = True

        return optimal_path

    def canny_edge_detection(self, image, low_threshold, high_threshold):
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges

    def create_closed_contour(self, optimal_path):
        closed_contour = []
        for point in optimal_path:
            closed_contour.append(point)
        return closed_contour

    # # Hiển thị kết quả trên giao diện
    # def display_result(self, image, closed_contour, ax):
    #     ax.clear()
    #     ax.imshow(image, cmap='gray')
    #     xs, ys = zip(*closed_contour)  # Tách x và y
    #     ax.plot(ys, xs, color='red', linewidth=2)
    #     ax.axis('off')

    # # Bước 6: Hiển thị kết quả
    def display_result(self, image, edges):
        plt.imshow(image, cmap='gray')
        xs, ys = np.nonzero(edges)  # Lấy tọa độ của các điểm biên
        plt.scatter(ys, xs, color='red', s=1)  # Vẽ các điểm biên lên ảnh gốc
        plt.axis('off')
        plt.show()

    # # Hàm xử lý ảnh và hiển thị kết quả
    # def process_image(self, path, ax):
    #     image = self.load_image(path)
    #     gradient = self.compute_gradient(image)
    #     cost = self.build_cost_function(gradient)
    #     optimal_path = self.dynamic_programming_edge_detection(cost)
    #     closed_contour = self.create_closed_contour(optimal_path)
    #     self.display_result(image, closed_contour, ax)
    #     plt.show()

    # Hàm xử lý ảnh và hiển thị kết quả
    def process_image(self, path, ax):
        image = self.load_image(path)
        optimal_path = self.canny_edge_detection(image, 100, 200)
        self.display_result(image, optimal_path)
