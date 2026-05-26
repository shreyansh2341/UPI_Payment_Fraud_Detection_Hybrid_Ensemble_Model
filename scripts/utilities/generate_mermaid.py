import os
import base64
import zlib
import urllib.request
import urllib.error

def generate_mermaid_via_kroki(mermaid_code, output_path):
    print(f"Generating diagram: {output_path}")
    
    # Kroki encoding: zlib compress -> base64 url-safe
    compressed = zlib.compress(mermaid_code.encode('utf-8'), 9)
    encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')
    url = f"https://kroki.io/mermaid/png/{encoded}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        print(f"Success: {output_path}")
    except Exception as e:
        print(f"Failed to generate {output_path}: {e}")

def create_dfd_level_0(out_dir):
    code = """
graph LR
    User(UPI User) -->|Transaction Data| System((0<br>Hybrid Ensemble<br>Fraud Detection<br>System))
    System -->|Transaction Status| User
    System -->|Suspicious Activity Report| Bank(Bank / Financial Institution)
    
    style User fill:#lightblue
    style Bank fill:#lightblue
    style System fill:#lightgreen
    """
    generate_mermaid_via_kroki(code, os.path.join(out_dir, "dfd_level_0.png"))

def create_dfd_level_1(out_dir):
    code = """
graph TB
    User(UPI User) -->|Initiates Transaction| P1((1.0<br>Data Collection<br>& Validation))
    P1 -->|Store Raw Data| D1[(D1 Transaction Database)]
    D1 -->|Historical Behavior| P2((2.0<br>Feature Engineering))
    P1 -->|Validated Data| P2
    
    P2 -->|Engineered Features| P3((3.0<br>Unsupervised<br>Autoencoder))
    P2 -->|Engineered Features| P4((4.0<br>Supervised<br>XGBoost + RF))
    P3 -->|AE Reconstruction Error| P4
    
    P3 -->|Anomaly Score| P5((5.0<br>Decision Engine<br>& Alerting))
    P4 -->|Fraud Probability| P5
    
    P5 -->|Block/Allow Status| User
    """
    generate_mermaid_via_kroki(code, os.path.join(out_dir, "dfd_level_1.png"))

def create_dfd_level_2(out_dir):
    code = """
graph TB
    P2_1((2.1 Extract Base Features)) -->|Base features| P2_2((2.2 Calculate Velocity))
    P2_2 -->|18 Features| P3_1((3.1 Feature Scaling))
    
    P3_1 -->|Scaled Data| P3_2((3.2 Autoencoder Inference))
    P3_2 -->|Reconstructed Data| P3_3((3.3 Calculate MSE))
    
    P3_1 -->|18 Features| P4_1((4.1 XGBoost))
    P3_1 -->|18 Features| P4_2((4.2 Random Forest))
    
    P3_3 -->|MSE 19th Feature| P4_1
    P3_3 -->|MSE 19th Feature| P4_2
    
    P4_1 -->|XGB Prob| P4_3((4.3 Ensemble Average))
    P4_2 -->|RF Prob| P4_3
    """
    generate_mermaid_via_kroki(code, os.path.join(out_dir, "dfd_level_2.png"))

def create_flowchart(out_dir):
    code = """
flowchart TB
    Start([Start: New Transaction]) --> Extract[Extract 18 Engineered Features]
    Extract --> Scale[Scale Features using RobustScaler]
    Scale --> AE[Calculate Autoencoder Reconstruction Error]
    
    AE --> PathSplit{Parallel Paths}
    
    PathSplit --> PathA[Path A: Supervised Classification]
    PathSplit --> PathB[Path B: Unsupervised Anomaly Detection]
    
    PathA --> Ensemble[Predict Weighted Prob P = 0.5 XGB + 0.5 RF]
    
    Ensemble --> PCheck{Is P >= 0.77?}
    PathB --> AECheck{Is AE Err > Threshold<br>OR IForest Anomaly?}
    
    PCheck -- Yes --> Block[BLOCK Transaction<br>High Confidence Fraud]
    PCheck -- No --> AECheck
    
    AECheck -- Yes --> Review[FLAG for Human Review<br>Novel Fraud]
    AECheck -- No --> Allow[ALLOW Transaction<br>Legitimate]
    
    style Block fill:#ffcccc
    style Review fill:#ffeecc
    style Allow fill:#ccffcc
    """
    generate_mermaid_via_kroki(code, os.path.join(out_dir, "flowchart.png"))

def create_use_case(out_dir):
    code = """
flowchart LR
    User[UPI User] -->|Performs Transaction| System(Fraud System)
    Bank[Financial Institution] -->|Monitors| System
    System -->|Detects Anomaly / Block| Bank
    System -->|Alerts| User
    Admin[Data Scientist] -->|Updates Thresholds| System
    """
    generate_mermaid_via_kroki(code, os.path.join(out_dir, "use_case_diagram.png"))

if __name__ == "__main__":
    out_dir = r"c:\Users\Shreyansh Rai\OneDrive\Desktop\Fraud_Detection_Model_Paysim\Fraud_Detection_Model_Paysim_CC\docs\dig_report"
    os.makedirs(out_dir, exist_ok=True)
    
    # Remove old empty/bad files
    for file in os.listdir(out_dir):
        if file.startswith("dfd") or file.startswith("flowchart") or file.startswith("use_case"):
            try:
                os.remove(os.path.join(out_dir, file))
            except Exception:
                pass
                
    create_dfd_level_0(out_dir)
    create_dfd_level_1(out_dir)
    create_dfd_level_2(out_dir)
    create_flowchart(out_dir)
    create_use_case(out_dir)
