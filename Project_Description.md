Project: Tri-Level Coordinated Dispatch for Active Distribution Networks

(Điều độ Phối hợp 3 Cấp cho Lưới điện Phân phối Chủ động: Hướng tiếp cận Lai giữa Tối ưu và Học sâu)

1. Executive Summary (Tóm tắt Dự án)

Dự án này phát triển một khung điều khiển phân cấp (Hierarchical Framework) mới cho sự tương tác giữa Nhà vận hành lưới phân phối (DSO) và Nhà máy điện ảo (VPP).

Trong bối cảnh lưới điện hiện đại với sự tham gia sâu của các nguồn năng lượng phân tán (DERs), nảy sinh mâu thuẫn cơ bản: VPP muốn tối đa hóa lợi nhuận bằng cách phản ứng với giá thị trường, trong khi DSO phải đảm bảo an toàn vật lý (điện áp, quá tải) trong điều kiện cấu trúc lưới thay đổi liên tục.

Giải pháp đề xuất sử dụng kiến trúc 3 lớp "DSO-Centric":

Layer 0 (DSO): Kiến tạo thị trường và lưới điện sử dụng Tối ưu hóa lồi (MISOCP).

Layer 1 (Aggregator): Lập lịch kinh tế sử dụng Tối ưu hóa ngẫu nhiên (Stochastic Optimization).

Layer 2 (Local Control): Điều khiển thời gian thực thích nghi sử dụng Graph Neural Networks (GNN).

2. Problem Statement (Vấn đề Nghiên cứu)

Các mô hình quản lý VPP hiện tại tồn tại 3 khoảng trống nghiên cứu (Research Gaps) lớn:

Topology Blindness (Sự mù mờ về cấu trúc):

Các VPP Aggregator thường mô hình hóa lưới điện như một nút đơn (Copper plate), bỏ qua các ràng buộc vật lý.

Hậu quả: Các lệnh điều độ từ VPP thường không khả thi (infeasible) khi lưới bị nghẽn hoặc tái cấu trúc, gây rủi ro an ninh.

Static & Uniform Pricing (Định giá tĩnh và đồng nhất):

Giá điện thường được giả định là ngoại sinh hoặc đồng nhất toàn vùng.

Hậu quả: Không phản ánh được sự khan hiếm cục bộ (Local Scarcity). VPP không có động lực kinh tế để hỗ trợ các vùng yếu của lưới.

Computational Scalability vs. Optimality (Mâu thuẫn Tối ưu - Tốc độ):

Các phương pháp tối ưu hóa (OPF/MPC) quá chậm cho điều khiển thời gian thực (<1s).

Các phương pháp RL thuần túy thiếu sự đảm bảo về tính tối ưu kinh tế toàn cục.

3. System Architecture (Kiến trúc Hệ thống)

Hệ thống hoạt động theo cơ chế Top-down Dispatch và Decoupled Responsibilities (Tách biệt trách nhiệm).

Layer 0: The Distribution System Operator (DSO)

Vai trò: "Market Maker & Grid Operator" (Người kiến tạo sân chơi).

Chức năng:

Dynamic Reconfiguration: Thay đổi trạng thái đóng/cắt của các switch để tối ưu hóa trào lưu công suất.

Zonal Pricing: Tính toán giá Năng lượng và Dự phòng cho từng vùng dựa trên tình trạng tắc nghẽn.

Đầu ra: Cấu trúc lưới ($A_t$) và Vector giá ($\Lambda_t$).

Layer 1: The VPP Aggregator

Vai trò: "Profit Maximizer" (Người chơi kinh tế).

Chức năng:

Nhận tín hiệu giá từ DSO.

Giải bài toán tối ưu hóa lợi nhuận trên các kịch bản giá dự báo.

Đặc điểm: "Topology-Agnostic" - Không xét đến trào lưu công suất lưới điện để giảm tải tính toán.

Đầu ra: Lệnh công suất tham chiếu ($P^*_{ref}$).

Layer 2: The Local Controller

Vai trò: "Grid Guardian" (Người bảo vệ lưới).

Chức năng:

Thực thi lệnh $P^*_{ref}$ từ Layer 1.

Sử dụng GNN để nhận thức cấu trúc lưới ($A_t$) từ Layer 0.

Tự động cắt giảm (Curtail) hoặc điều chỉnh công suất phản kháng nếu phát hiện nguy cơ vi phạm điện áp.

Công nghệ: GNN-Enhanced Multi-Agent Reinforcement Learning (MARL).

4. Methodology & Mathematical Formulation

4.1. Layer 0: Co-Optimized Reconfiguration & Pricing (MISOCP)

DSO giải bài toán tối ưu hóa hỗn hợp nguyên (Mixed-Integer) để tìm cấu hình lưới tối ưu.

Hàm mục tiêu:


$$\min \sum_{t} \left( C_{Loss} P_{Loss,t} + C_{Switch} \sum_{l} |\alpha_{l,t} - \alpha_{l,t-1}| + C_{Unbal} V_{idx,t} \right)$$

Cơ chế Định giá Zonal (Zonal Pricing Mechanism):
Sau khi cố định các biến nhị phân (Topology), DSO giải bài toán tuyến tính để lấy Shadow Prices:

Giá Năng lượng ($\lambda^{En}_{z,t}$): Bình quân gia quyền theo tải của giá nút (DLMP).


$$\lambda^{En}_{z,t} = \frac{\sum_{i \in z} P_{load,i} \cdot \lambda^{DLMP}_{i,t}}{\sum_{i \in z} P_{load,i}}$$

Giá Dự phòng ($\lambda^{Res}_{z,t}$): Dựa trên ràng buộc an ninh (Contingency Constraints).


$$\lambda^{Res}_{z,t} = \mu_{sys,t} + \eta_{z,t}$$

$\mu_{sys,t}$: Giá dự phòng cơ sở toàn hệ thống.

$\eta_{z,t}$: Phụ phí khan hiếm (Scarcity Premium) nếu Zone $z$ bị nghẽn/rủi ro.

4.2. Layer 1: Stochastic Bidding Strategy

VPP tối ưu hóa lợi nhuận kỳ vọng trên cây kịch bản giá (Price Scenarios) $\Omega$.

Hàm mục tiêu:


$$\max \sum_{\omega \in \Omega} \pi_\omega \sum_{t=1}^{T} \sum_{z \in \mathcal{Z}} \left[ \lambda^{En}_{z,t,\omega} P_{inj, z,t} + \lambda^{Res}_{z,t,\omega} R_{z,t} - C_{deg}(P_{inj}) \right]$$

Ràng buộc:

Cân bằng năng lượng Pin ảo (Virtual Battery): $SoC_{t+1} = SoC_t + P \Delta t$.

Giới hạn công suất Inverter tổng.

Lưu ý: Loại bỏ hoàn toàn ràng buộc Power Flow.

4.3. Layer 2: GNN-Based Real-Time Control

Agent $i$ điều khiển Inverter tại nút $i$.

Đầu vào (Observation Space):

Local: $SoC_i, P_{load,i}, V_i$.

Global Command: $P^*_{ref}$ (Tổng công suất yêu cầu từ Layer 1).

Topological Feature: Ma trận kề $A_t$ (Input cho lớp Graph Convolution).

Hàm thưởng (Reward Function):


$$R_t = \underbrace{-\alpha (P_{\Sigma} - P^*_{ref})^2}_{\text{Tracking Error}} - \underbrace{\beta \sum_{j \in \mathcal{N}(i)} \text{relu}(|V_j - 1| - \epsilon)}_{\text{Voltage Violation}} + \underbrace{\gamma \cdot \lambda^{Res}_{z,t} \cdot R_{avail}}_{\text{Reserve Bonus}}$$

Kiến trúc GNN:
Sử dụng Graph Attention Network (GAT) để Agent học được trọng số ảnh hưởng của các nút lân cận, từ đó tự động nhận biết nút nào đang là "nút cổ chai" trong cấu trúc lưới hiện tại.

5. Implementation Strategy (Chiến lược Thực hiện)

A. Tech Stack

Component

Technology

Role

Grid Simulation

Pandapower

Mô phỏng lưới điện, tính trào lưu công suất, thay đổi switch.

Optimization

Pyomo + Gurobi

Giải bài toán MISOCP (Layer 0) và Stochastic Opt (Layer 1).

AI Core

PyTorch Geometric

Xây dựng mạng GNN xử lý đồ thị động.

RL Framework

Ray RLLib

Huấn luyện Multi-Agent (MAPPO) song song.

B. Data Processing

Grid Data: Lưới IEEE 123-bus đã được chuyển đổi sang JSON (Single-phase equivalent).

Modification: Đã thêm các Tie-lines (Switch thường mở) để cho phép bài toán Reconfiguration hoạt động.

Load Profiles: Dữ liệu tải thực tế (scaled).

Price Scenarios: Được sinh ngẫu nhiên dựa trên các sự kiện tắc nghẽn giả định (Synthetic Scenarios).

6. Key Contributions (Đóng góp Chính)

Integrated Market-Physics Framework: Đề xuất mô hình điều khiển 3 cấp tích hợp chặt chẽ, nơi Giá (Kinh tế) và Cấu trúc lưới (Vật lý) là các biến nội sinh được tính toán tối ưu, thay vì là tham số cố định.

Contingency-based Zonal Pricing: Áp dụng cơ chế định giá dự phòng dựa trên rủi ro, tạo tín hiệu kinh tế chính xác để điều hướng hành vi của VPP hỗ trợ lưới điện.

Topology-Adaptive AI Control: Chứng minh khả năng Zero-shot Generalization của GNN. Hệ thống duy trì ổn định điện áp ngay cả khi DSO thay đổi cấu trúc lưới đột ngột, khắc phục điểm yếu "học vẹt" của các mô hình RL truyền thống.