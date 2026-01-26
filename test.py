import pandapower as pp
import pandas as pd
import numpy as np

def validate_export(json_path):
    print(f"--- CHECKING FILE: {json_path} ---")
    
    # 1. Thử load file
    try:
        net = pp.from_json(json_path)
        print("OK: Load file successful")
    except Exception as e:
        print(f"ERROR: Load file failed: {e}")
        return

    # 2. Thống kê cơ bản
    print("\n--- SYSTEM SUMMARY ---")
    print(f"Bus count: {len(net.bus)}")
    print(f"Load count: {len(net.load)}")
    print(f"Line count: {len(net.line)}")
    print(f"Switch count: {len(net.switch)}")
    print(f"Gen/Ext_Grid count: {len(net.gen) + len(net.ext_grid)}")

    # 3. Kiểm tra chuyển đổi Switch (QUAN TRỌNG)
    # Kiểm tra xem còn đường dây nào có trở kháng siêu nhỏ không
    # Ngưỡng kiểm tra: R < 1e-4 và X < 1e-4
    suspicious_lines = net.line[(net.line.r_ohm_per_km < 1e-4) & (net.line.x_ohm_per_km < 1e-4)]
    
    if len(suspicious_lines) > 0:
        print(f"\nWARN: Remaining {len(suspicious_lines)} near-zero lines not converted to switches")
        print(suspicious_lines[['from_bus', 'to_bus', 'r_ohm_per_km', 'x_ohm_per_km']].head())
    else:
        print("\nOK: Switch check passed (no near-zero lines)")

    # 4. Kiểm tra tính liên thông (Connectivity)
    # Dùng đồ thị NetworkX để xem có nút nào bị cô lập không
    mg = pp.topology.create_nxgraph(net, include_lines=True, include_switches=True)
    # Số thành phần liên thông
    n_components = 0
    # NetworkX 2.x trở lên dùng connected_components
    try:
        import networkx as nx
        n_components = nx.number_connected_components(mg)
    except:
        pass # Bỏ qua nếu lỗi thư viện
        
    print("\n--- CONNECTIVITY ---")
    if n_components == 1:
         print("OK: Network is fully connected (1 island)")
    else:
         print(f"WARN: Network split into {n_components} islands. Check switch status.")

    # 5. Tie-line check (reconfiguration)
    open_switches = net.switch[net.switch.closed == False]
    print("\n--- TIE-LINES ---")
    if len(open_switches) > 0:
        print(f"OK: Found {len(open_switches)} open tie-switches")
    else:
        print("WARN: No open tie-switches found")

if __name__ == "__main__":
    validate_export("ieee123_optimized.json")