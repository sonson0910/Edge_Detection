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
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return gradient_magnitude

    # Bước 3: Xây dựng hàm chi phí dựa trên gradient
    def build_cost_function(self, gradient):
        cost = -gradient  # Chi phí dựa trên độ lớn gradient
        return cost

    # Áp dụng quy hoạch động để dò biên và bao trọn vật thể
    def dynamic_programming_edge_detection(self, cost):
        rows, cols = cost.shape
        dp = np.full_like(cost, float('inf'))  # Ma trận lưu chi phí tối thiểu
        path = np.zeros((rows, cols, 2), dtype=int)  # Ma trận lưu hướng đi: 2 chiều để lưu tọa độ (di, dj)
        visited = np.zeros_like(cost, dtype=bool)  # Ma trận lưu các điểm đã đi qua

        # Hướng di chuyển: lên, xuống, trái, phải, chéo trên-trái, chéo trên-phải, chéo dưới-trái, chéo dưới-phải
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Khởi tạo hàng đầu tiên của ma trận DP bằng chi phí của hàng đầu tiên
        dp[0, :] = cost[0, :]

        # Duyệt qua các hàng và cột để tính chi phí tối ưu
        for i in range(1, rows):
            for j in range(cols):
                min_cost = float('inf')
                best_direction = None

                # Duyệt qua các hướng di chuyển
                for di, dj in directions:
                    new_i, new_j = i + di, j + dj

                    # Kiểm tra điểm có nằm trong ma trận không
                    if 0 <= new_i < rows and 0 <= new_j < cols:
                        # Tìm chi phí tối thiểu từ các hướng
                        if dp[new_i, new_j] < min_cost:
                            min_cost = dp[new_i, new_j]
                            best_direction = (di, dj)

                # Cập nhật bảng DP và lưu hướng đi
                dp[i, j] = cost[i, j] + min_cost
                if best_direction is not None:
                    path[i, j] = best_direction

        # Truy ngược lại đường đi tối ưu để bao trọn thực thể
        optimal_path = []
        min_index = np.argmin(dp[-1, :])  # Lấy vị trí có chi phí nhỏ nhất ở hàng cuối cùng
        optimal_path.append((rows - 1, min_index))  # Thêm điểm cuối cùng vào đường đi
        visited[rows - 1, min_index] = True  # Đánh dấu điểm đã đi qua

        current_row, current_col = rows - 1, min_index

        # Truy ngược lại cho đến khi đi hết vòng bao quanh thực thể
        while True:
            di, dj = path[current_row, current_col]
            next_row = current_row + di
            next_col = current_col + dj

            # Nếu điểm tiếp theo đã được truy ngược qua (vòng hoàn tất), thì thoát
            if visited[next_row, next_col]:
                break

            # Ưu tiên di chuyển sang trái nếu trong cùng một hàng
            if di == 0 and dj == -1 and current_col > 0:
                optimal_path.append((current_row, current_col - 1))
                visited[current_row, current_col - 1] = True
                current_col -= 1
            else:
                optimal_path.append((next_row, next_col))
                visited[next_row, next_col] = True
                current_row, current_col = next_row, next_col

        return optimal_path

    # Bước 4-2: Phát hiện cạnh sử dụng thuật toán Canny
    def canny_edge_detection(self, image, low_threshold, high_threshold):
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges

    # Bước 5: Tạo đường bao khép kín
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
