import cv2
from baselines.downsample_baseline import downsample_baseline
from simulator.phosphene_simulator import render_phosphene

image_path = "E:/Semi Study/CLC 12/Normal Pictures/Fanzhendong vs zhangben quarter final.jpg"
image = cv2.imread(image_path)
baseline = downsample_baseline(image)
phosphene = render_phosphene(image)
cv2.imshow("Original", image)
cv2.imshow("Downsample Baseline", baseline)
cv2.imshow("Phosphene Simulator", phosphene)
cv2.imwrite("results/baseline1.png", baseline)
cv2.imwrite("results/phosphene1.png", phosphene)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Output using this in the terminal instead of simply running the file: 
python -m demo.compare_baseline_vs_phosphene
'''
