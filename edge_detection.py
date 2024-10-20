import cv2
import numpy as np
import matplotlib.pyplot as plt


class EdgeDetection:
    # Bước 1: Đọc và chuyển đổi ảnh sang grayscale
    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image

    # Bước 2: Tính gradient của ảnh
    def compute_gradient(self, image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return gradient_magnitude

    # Bước 3: Xây dựng hàm chi phí dựa trên gradient
    def build_cost_function(self, gradient):
        cost = -gradient  # Chi phí dựa trên độ lớn gradient
        return cost

    # Bước 4: Tìm đường biên tối ưu bằng quy hoạch động
    def dynamic_programming_edge_detection(self, cost):
        rows, cols = cost.shape
        dp = np.zeros_like(cost)  # Bảng chi phí
        path = np.zeros((rows, cols), dtype=int)  # Lưu hướng đi

        # Khởi tạo bảng chi phí cho hàng đầu tiên
        dp[0, :] = cost[0, :]

        # Duyệt qua các hàng tiếp theo
        for i in range(1, rows):
            for j in range(cols):
                if j == 0:
                    min_cost = min(dp[i - 1, j], dp[i - 1, j + 1])
                    path[i, j] = 0 if min_cost == dp[i - 1, j] else 1
                elif j == cols - 1:
                    min_cost = min(dp[i - 1, j - 1], dp[i - 1, j])
                    path[i, j] = -1 if min_cost == dp[i - 1, j - 1] else 0
                else:
                    min_cost = min(dp[i - 1, j - 1], dp[i - 1, j], dp[i - 1, j + 1])
                    if min_cost == dp[i - 1, j - 1]:
                        path[i, j] = -1  # Diagonal left
                    elif min_cost == dp[i - 1, j]:
                        path[i, j] = 0   # Up
                    else:
                        path[i, j] = 1   # Diagonal right
                dp[i, j] = cost[i, j] + min_cost

        min_index = np.argmin(dp[-1, :])
        optimal_path = [(rows - 1, min_index)]

        for i in range(rows - 1, 0, -1):
            direction = path[i, optimal_path[-1][1]]
            new_index = optimal_path[-1][1] + direction
            if new_index < 0:
                new_index = 0
            elif new_index >= cols:
                new_index = cols - 1
            optimal_path.append((i - 1, new_index))

        optimal_path.reverse()
        return optimal_path

    # Bước 5: Tạo đường bao khép kín
    def create_closed_contour(self, optimal_path):
        closed_contour = []
        for point in optimal_path:
            closed_contour.append(point)
        return closed_contour

    # Hiển thị kết quả trên giao diện
    def display_result(self, image, closed_contour, ax):
        ax.clear()
        ax.imshow(image, cmap='gray')
        xs, ys = zip(*closed_contour)  # Tách x và y
        ax.plot(ys, xs, color='red', linewidth=2)
        ax.axis('off')

    # Hàm xử lý ảnh và hiển thị kết quả
    def process_image(self, path, ax):
        image = self.load_image(path)
        gradient = self.compute_gradient(image)
        cost = self.build_cost_function(gradient)
        optimal_path = self.dynamic_programming_edge_detection(cost)
        closed_contour = self.create_closed_contour(optimal_path)
        self.display_result(image, closed_contour, ax)
        plt.show()