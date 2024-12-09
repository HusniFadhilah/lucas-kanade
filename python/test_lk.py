import numpy as np,os
import helper,time,logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade
from file_utils import mkdir_if_missing

data_name = 'car2'      # could choose from (car1, car2, landing) 
# Configure logging with dataset-specific file in helper.resultdir
log_path = f"{helper.resultdir}/performance_logs"
mkdir_if_missing(log_path)  # Ensure the log directory exists
log_file = os.path.join(log_path, f"lk/{data_name}_performance_log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# Log the dataset being processed
logging.info(f"Processing dataset: {data_name}")
print(f"Processing dataset: {data_name}")
# load data name
data = np.load(f'{helper.datadir}/{data_name}.npy')

# obtain the initial rect with format (x1, y1, x2, y2)
if data_name == 'car1':
    initial = np.array([170, 130, 290, 250])   
elif data_name == 'car2':
    initial = np.array([59,116,145,151])    
elif data_name == 'landing':
    initial = np.array([440, 80, 560, 140])     
else:
    assert False, 'the data name must be one of (car1, car2, landing)'

numFrames = data.shape[2]
w = initial[2] - initial[0]
h = initial[3] - initial[1]

# loop over frames
rects = []
rects.append(initial)
fig = plt.figure(1)
ax = fig.add_subplot(111)

# Track total processing time
total_start_time = time.time()
frame_times = []
for i in range(numFrames-1):
    frame_start_time = time.time()
    print("frame****************", i)
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    rect = rects[i]

    # run algorithm
    dx, dy,avg_pixel_diff = LucasKanade(It, It1, rect)
    logging.info(f"Frame {i}, Affine Matrix M: {dx,dy}")
    logging.info(f"Frame {i}, Average Pixel Difference: {avg_pixel_diff:.4f}")

    # transform the old rect to new one
    newRect = np.array([rect[0] + dx, rect[1] + dy, rect[0] + dx + w, rect[1] + dy + h])
    rects.append(newRect)

    # Show image
    # print("Plotting: ", rect)
    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    plt.imshow(It1, cmap='gray')
    save_path = f"{helper.resultdir}/lk/{data_name}/frame{i+1:06d}.jpg"
    mkdir_if_missing(save_path)
    plt.savefig(save_path)
    plt.pause(0.01)
    ax.clear()

    # End timing for the frame
    frame_end_time = time.time()
    frame_duration = frame_end_time - frame_start_time
    frame_times.append(frame_duration)
    logging.info(f"Frame {i}, Processing Time: {frame_duration:.4f} seconds")

# End total processing time
total_end_time = time.time()
total_duration = total_end_time - total_start_time

logging.info(f"Dataset: {data_name}, Total Processing Time: {total_duration:.4f} seconds")
logging.info(f"Dataset: {data_name}, Average Time Per Frame: {np.mean(frame_times):.4f} seconds")
print(f"Dataset: {data_name} processed in {total_duration:.4f} seconds.")
print(f"Average time per frame: {np.mean(frame_times):.4f} seconds.")

print(f"Total processing time: {total_duration:.4f} seconds.")
print(f"Average time per frame: {np.mean(frame_times):.4f} seconds.")
