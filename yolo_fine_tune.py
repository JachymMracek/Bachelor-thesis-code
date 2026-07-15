from ultralytics import YOLO
import argparse

# SOURCES
# https://docs.ultralytics.com/guides/finetuning-guide#key-hyperparameters-for-fine-tuning

################################################################################
################################################################################
########################### YOLO fine tuning script ############################
################################################################################
################################################################################


argumentParser = argparse.ArgumentParser()
argumentParser.add_argument("--yaml_yolo_path",default=r"D:\hit_dataset\data.yaml", help="Please, write your yolo dataset yaml path.")
argumentParser.add_argument("--results_training_path",default="yolo_hit_ball", help="Please, write name where the results of training will be saved")

# train yolo
def train_yolo(yaml_yolo_path,results_training_path,YOLO_MODEL_TYPE = "yolo26n.pt"):
    
    model = YOLO(YOLO_MODEL_TYPE)

    model.train(data=yaml_yolo_path,epochs=250,name=results_training_path, exist_ok=False)

def main():
    args = argumentParser.parse_args()
    
    train_yolo(args.yaml_yolo_path,args.results_training_path)

if __name__ == "__main__":
    main()