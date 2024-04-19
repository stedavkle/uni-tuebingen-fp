import numpy as np
import random
import matplotlib.pyplot as plt
import time

def generate_random_walks(n_walks, steps_per_walk, max_step_size=1024):
    walks = []
    timestamps = []
    for _ in range(n_walks):
        start_position = random.randint(-32768, 0)
        end_position = random.randint(0, 32768)

        current_position = start_position
        
        walk = [start_position]
        timestamp_walk = [time.time()]
        
        for _ in range(steps_per_walk - 1):
            # Calculate the remaining distance to the target
            distance = end_position - current_position
            
            # Determine the step size (not exceeding max_step_size and remaining distance)
            step_size = random.randint(1, min(max_step_size, abs(distance)))
            step = step_size if distance >= 0 else -step_size
            # there shold be a 0.3 probability that the stepper motor will move in the opposite direction
            if random.random() < 0.4:
                step = -step_size


            # Update the current position
            current_position += step
            walk.append(current_position)
            timestamp_walk.append(timestamp_walk[-1] + 1)  # one second interval for each step
            
            # Check if we are close enough to the final position
            if abs(end_position - current_position) < step_size:
                break
        
        # Ensure the final step takes us exactly to the end_position
        walk.append(end_position)
        timestamp_walk.append(timestamp_walk[-1] + 1)

        walk = np.convolve(walk, np.ones(32)/32, mode='valid').tolist()
        timestamp_walk = timestamp_walk[:len(walk)]

        current_position = walk[-1]
        # now use a logaritmic function to approach the final position in the next 100 steps
        for i in range(200):
            current_position = int((end_position - current_position) * (1 - np.exp(-i/200)) + current_position)
            walk.append(current_position)
            timestamp_walk.append(timestamp_walk[-1] + 1)
            if abs(end_position - current_position) < step_size/100:
                break
        
        

        walks.append(walk)
        timestamps.append(timestamp_walk)
    
    return walks, timestamps

# Generate 3 random walks with at least 360 steps each
n_walks = 15  # number of random walks
steps_per_walk = 1000  # steps in each walk
walks, timestamps = generate_random_walks(n_walks, steps_per_walk, max_step_size=500)

longest_walk = max(len(walk) for walk in walks)
steps_per_min = longest_walk / 5

# Plot the walks
plt.figure(figsize=(12, 6))
for i in range(n_walks):
    plt.plot(timestamps[i], walks[i], label=f'Walk {i+1}')
plt.title('Random Walks of Stepper Motor Position')
plt.xlabel('Time (seconds)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()
