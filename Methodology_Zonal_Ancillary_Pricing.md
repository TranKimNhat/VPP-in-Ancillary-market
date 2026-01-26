Integration Strategy for Layer 0: Reconfiguration & Pricing

(Chiến lược Tích hợp Bài toán Tái cấu trúc vào Layer 0)

1. Đánh giá Nguồn tài liệu

Paper A: Reconfiguration of 123-bus... (Nacar et al.)

Đóng góp: Đề xuất hàm mục tiêu giảm thiểu mất cân bằng pha (Unbalance Index).

Hạn chế: Sử dụng giải thuật Metaheuristic (Slime Mould). Không tạo ra Shadow Price.

Quyết định: Chỉ lấy Hàm mục tiêu, không lấy Giải thuật.

Paper B: Topology and Reactive Power Co-Optimization... (Mohammadi Vaniar et al.)

Đóng góp: Khung đồng tối ưu hóa (Co-optimization) giữa Tái cấu trúc và Công suất phản kháng.

Ưu điểm: Phù hợp với bài toán ổn định điện áp của DSO.

Quyết định: Áp dụng Khung bài toán (Formulation) này.

2. Mô hình Toán học Đề xuất (Proposed Layer 0 Formulation)

Để đảm bảo tính được Zonal Pricing (yêu cầu bắt buộc của đồ án), chúng ta phải chuyển đổi các ý tưởng trên về dạng MISOCP (Mixed-Integer Second-Order Cone Programming).

2.1. Hàm Mục tiêu (Objective Function)

Kết hợp ý tưởng giảm tổn thất của Paper A và điều phối Q của Paper B:

$$\min \sum_{t} \left( \underbrace{C_{Loss} \cdot P_{Loss, t}}_{\text{Từ Paper A}} + \underbrace{C_{Switch} \cdot \sum_{l} |\alpha_{l,t} - \alpha_{l,t-1}|}_{\text{Chi phí hao mòn}} + \underbrace{C_{Unbalance} \cdot V_{dev, t}}_{\text{Xấp xỉ Unbalance Index}} \right)$$

2.2. Các Ràng buộc Cốt lõi (Core Constraints)

A. Ràng buộc Tái cấu trúc (Reconfiguration)

Sử dụng biến nhị phân $\alpha_{l,t}$ (1: đóng, 0: mở).

Radial Constraint: $\sum \alpha_{l,t} = N_{bus} - 1$.

Power Flow (Big-M): Áp dụng kỹ thuật Big-M để ngắt dòng chảy khi $\alpha_{l,t}=0$.

B. Ràng buộc Công suất Phản kháng (Reactive Power - Từ Paper B)

DSO điều phối cả tụ bù và Inverter để hỗ trợ áp:

$Q_{min} \le Q_{i,t} \le Q_{max}$.

Ràng buộc này giúp nới lỏng giới hạn điện áp, giúp bài toán Reconfiguration dễ tìm nghiệm hơn (Feasibility).

3. Quy trình Tính toán Giá (Pricing Workflow)

Đây là bước quan trọng để tránh "bẫy" Metaheuristic:

Bước 1 (Integer Solve): Giải bài toán MISOCP đầy đủ (với biến nhị phân $\alpha$) để tìm cấu trúc lưới tối ưu và trạng thái tụ bù.

Kết quả: Topo lưới mới ($A^*_t$).

Bước 2 (Continuous Solve - Pricing Run):

Cố định tất cả biến nhị phân $\alpha$ theo kết quả Bước 1.

Giải lại bài toán dưới dạng SOCP (Lồi).

Trích xuất Biến đối ngẫu (Dual Variables):

$\lambda_{node}$ (Shadow price của cân bằng công suất).

$\mu_{voltage}$ (Shadow price của ràng buộc điện áp).

Bước 3 (Zonal Aggregation):

Tính toán Zonal Energy Price và Zonal Reserve Price từ các biến đối ngẫu trên (theo file Methodology_Zonal_Ancillary_Pricing.md trước đó).

4. Kết luận

Bạn hoàn toàn có thể áp dụng ý tưởng của 2 bài báo này.

Key Modification: Phải mô hình hóa lại dưới dạng MISOCP (sử dụng thư viện Gurobi/Pyomo) thay vì dùng Slime Mould Algorithm hay Genetic Algorithm để đảm bảo tính nhất quán với cơ chế định giá thị trường.