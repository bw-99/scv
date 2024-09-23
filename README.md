SCV: Selectively Crossing Vectors for Accelerated Model Training


https://www.notion.so/SCV-Selectively-Crossing-Vectors-for-Accelerated-Model-Training-101d2f288baf80b18531c9739608d9ef

docker build -t vision_dcn_env .

docker run -dit --name con --shm-size=10g -v ~/visionDCN:/workspace  --runtime=nvidia vision_dcn_env

docker run -dit --name con --shm-size=10g -v ~/visionDCN:/workspace nvcr.io/nvidia/pytorch:22.12-py3


docker run -dit --name bw_con --shm-size=10g -v /home/lab05/bw/scv:/workspace vision_dcn_env
https://github.com/reczoo/Datasets