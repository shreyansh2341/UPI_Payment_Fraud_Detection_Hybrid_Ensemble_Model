import os
from graphviz import Digraph

def create_dfd_level_0(output_dir):
    dot = Digraph("DFD_Level_0", format='png')
    dot.attr(rankdir='LR')
    dot.attr(dpi='300')
    
    # External Entities
    dot.node("User", "UPI User", shape="box", style="filled", fillcolor="lightblue")
    dot.node("Bank", "Bank / Financial\nInstitution", shape="box", style="filled", fillcolor="lightblue")
    
    # Main Process
    dot.node("System", "0\nHybrid Ensemble\nFraud Detection\nSystem", shape="circle", style="filled", fillcolor="lightgreen")
    
    # Flows
    dot.edge("User", "System", "Transaction Data\n(Amount, Time, Account Info)")
    dot.edge("System", "User", "Transaction Status\n(Allowed / Blocked)")
    dot.edge("System", "Bank", "Suspicious Activity Report\n(Fraud Alert / Review Needed)")
    
    dot.render(os.path.join(output_dir, "dfd_level_0"))
    print("Generated DFD Level 0")

def create_dfd_level_1(output_dir):
    dot = Digraph("DFD_Level_1", format='png')
    dot.attr(rankdir='TB')
    dot.attr(dpi='300')
    
    # External
    dot.node("User", "UPI User", shape="box", style="filled", fillcolor="lightblue")
    
    # Processes
    dot.node("P1", "1.0\nData Collection\n& Validation", shape="circle")
    dot.node("P2", "2.0\nFeature Engineering\n(Velocity & Ratio)", shape="circle")
    dot.node("P3", "3.0\nUnsupervised Anomaly\nDetection (Autoencoder)", shape="circle")
    dot.node("P4", "4.0\nSupervised Classification\n(XGBoost + RF)", shape="circle")
    dot.node("P5", "5.0\nDecision Engine & Alerting", shape="circle")
    
    # Data Stores
    dot.node("D1", "D1 Transaction Database", shape="cylinder")
    
    # Flows
    dot.edge("User", "P1", "Initiates Transaction")
    dot.edge("P1", "D1", "Store Raw Data")
    dot.edge("D1", "P2", "Historical Behavior")
    dot.edge("P1", "P2", "Validated Data")
    
    dot.edge("P2", "P3", "Engineered Features (18)")
    dot.edge("P3", "P4", "AE Reconstruction Error\n(19th Feature)")
    dot.edge("P2", "P4", "Engineered Features (18)")
    
    dot.edge("P3", "P5", "Anomaly Score (Threshold > 0.052)")
    dot.edge("P4", "P5", "Fraud Probability (Confidence)")
    
    dot.edge("P5", "User", "Block/Allow Status")
    
    dot.render(os.path.join(output_dir, "dfd_level_1"))
    print("Generated DFD Level 1")

def create_dfd_level_2(output_dir):
    dot = Digraph("DFD_Level_2", format='png')
    dot.attr(rankdir='TB')
    dot.attr(dpi='300')
    
    dot.node("P2_1", "2.1 Extract Base Features\n(Amount, Time, Balance)", shape="circle")
    dot.node("P2_2", "2.2 Calculate Velocity\n(Cumulative stats, time since last)", shape="circle")
    
    dot.node("P3_1", "3.1 Feature Scaling\n(RobustScaler)", shape="circle")
    dot.node("P3_2", "3.2 Autoencoder Inference", shape="circle")
    dot.node("P3_3", "3.3 Calculate MSE", shape="circle")
    
    dot.node("P4_1", "4.1 XGBoost Prediction", shape="circle")
    dot.node("P4_2", "4.2 Random Forest Prediction", shape="circle")
    dot.node("P4_3", "4.3 Ensemble Average", shape="circle")
    
    dot.edge("P2_1", "P2_2", "Base features")
    dot.edge("P2_2", "P3_1", "18 Features")
    
    dot.edge("P3_1", "P3_2", "Scaled Data")
    dot.edge("P3_2", "P3_3", "Reconstructed Data")
    
    dot.edge("P3_3", "P4_1", "MSE (19th Feature)")
    dot.edge("P3_3", "P4_2", "MSE (19th Feature)")
    dot.edge("P3_1", "P4_1", "18 Features")
    dot.edge("P3_1", "P4_2", "18 Features")
    
    dot.edge("P4_1", "P4_3", "XGB Prob")
    dot.edge("P4_2", "P4_3", "RF Prob")
    
    dot.render(os.path.join(output_dir, "dfd_level_2"))
    print("Generated DFD Level 2")

def create_flowchart(output_dir):
    dot = Digraph("Flowchart", format='png')
    dot.attr(rankdir='TB')
    dot.attr(dpi='300')
    
    dot.node("Start", "Start: New Transaction", shape="oval")
    dot.node("Extract", "Extract 18 Engineered Features\n(Velocity, Time, Balance)", shape="box")
    dot.node("Scale", "Scale Features (RobustScaler)", shape="box")
    dot.node("AE", "Calculate Autoencoder\nReconstruction Error (AE_Err)", shape="box")
    
    dot.node("PathSplit", "Parallel Paths", shape="diamond")
    
    dot.node("PathA", "Path A: Supervised", shape="box", style="dashed")
    dot.node("PathB", "Path B: Unsupervised", shape="box", style="dashed")
    
    dot.node("Ensemble", "Predict Weighted Prob (P)\n0.5(XGB) + 0.5(RF)\nusing 19 Features", shape="box")
    dot.node("AE_Check", "Is AE_Err > Threshold\nor IForest Flags Anomaly?", shape="diamond")
    
    dot.node("P_Check", "Is P >= 0.77?", shape="diamond")
    
    dot.node("Block", "BLOCK Transaction\n(High Confidence Fraud)", shape="box", style="filled", fillcolor="Salmon")
    dot.node("Review", "FLAG for Human Review\n(Novel Fraud / Anomaly)", shape="box", style="filled", fillcolor="Orange")
    dot.node("Allow", "ALLOW Transaction\n(Legitimate)", shape="box", style="filled", fillcolor="lightgreen")
    
    dot.edge("Start", "Extract")
    dot.edge("Extract", "Scale")
    dot.edge("Scale", "AE")
    dot.edge("AE", "PathSplit")
    
    dot.edge("PathSplit", "PathA")
    dot.edge("PathSplit", "PathB")
    
    dot.edge("PathA", "Ensemble")
    dot.edge("Ensemble", "P_Check")
    
    dot.edge("PathB", "AE_Check")
    
    dot.edge("P_Check", "Block", "Yes")
    dot.edge("P_Check", "AE_Check", "No")
    
    dot.edge("AE_Check", "Review", "Yes")
    dot.edge("AE_Check", "Allow", "No")

    dot.render(os.path.join(output_dir, "flowchart"))
    print("Generated Flowchart")

if __name__ == "__main__":
    out_dir = r"c:\Users\Shreyansh Rai\OneDrive\Desktop\Fraud_Detection_Model_Paysim\Fraud_Detection_Model_Paysim_CC\docs\dig_report"
    os.makedirs(out_dir, exist_ok=True)
    
    create_dfd_level_0(out_dir)
    create_dfd_level_1(out_dir)
    create_dfd_level_2(out_dir)
    create_flowchart(out_dir)
