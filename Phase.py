from psychopy import prefs
prefs.hardware['audioLib'] = ['ptb', 'pygame', 'pyo', 'sounddevice']
prefs.hardware['audioDevice'] = ['Speakers (USBAudio2.0)']
from psychopy import visual, core, sound, monitors, gui
import numpy as np
import random, csv, os
from pybpod_rotaryencoder_module.module_api import RotaryEncoderModule
import matplotlib.pyplot as plt
import ast
from pathlib import Path
from datetime import date
import threading
import time
import subprocess
from datetime import datetime, timedelta
import sys
import statistics
from psychopy.hardware import keyboard
from psychopy import core
import math

# pink noise generator
def generate_pink_noise(duration, sample_rate=48000):
    n_samples = int(duration * sample_rate)
    uneven = n_samples % 2
    X = np.random.randn(n_samples // 2 + 1 + uneven) + 1j * np.random.randn(n_samples // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    y = y / np.max(np.abs(y))
    return y.astype(np.float32)

# Rotary Encoder
serial_port = 'COM4'
m = RotaryEncoderModule(serialport=serial_port)

#Experiment Ending Check
cond1_time = 70
cond2_time = cond1_time * 2
trial_num = 6 #determine how much trail number should stop
trialNum = 5 #rolling window trail number
totalRTNum = 0.0
totalRTArray = []
certainRT = []

def session_ending(totalReactionTime, trialNumber, currentMedian, certainReactionTimeM):
    if (totalReactionTime > cond1_time and trialNumber < trial_num) or (totalReactionTime > cond2_time and trialNumber > trial_num):
        return True
    elif trialNumber > trial_num and certainReactionTimeM > 5*currentMedian:
        return True
    return False

#visual stimuli
length = 1280
width = 800
radius = 250
mon = monitors.Monitor(name='testMonitor', width=53.0, distance=70.0)
mon.setSizePix((length, width))
win = visual.Window(size=(length, width), units="pix", fullscr=False, pos=(0, 0))
circle = visual.Circle(win, radius=radius, fillColor='white', lineColor=None)
kb = keyboard.Keyboard()
ITI_s = (1, 1.5)

#get mouse names from user
name_path = Path(r"Z:\16-Mouse-HRD\3-documentation\MouseRecord.csv")
name_rec = []
with open(name_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for name in reader:
                name_rec.append(name[0])

name_rec.append('New Mouse')
            
Name = {
    'Mouse name' : name_rec,
}
dlg = gui.DlgFromDict(dictionary=Name, title='Mouse Information')

if dlg.OK:
    mouseName = Name['Mouse name']
else:
    core.quit()

if mouseName == 'New Mouse':
    NewName = {
        'Name' : '',
    }
    dlg = gui.DlgFromDict(dictionary=NewName, title='Name')
    if dlg.OK:
        mouseName = NewName['Name']
    else:
        core.quit()
    with open(name_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([mouseName])
    
Info = {
    'time for stimuli in ms': 1000,
    'Number for 0 lag': 4,
    'Max duration in ms': 20000,
}
dialog_order = ['time for stimuli in ms', 'Number for 0 lag', 'Max duration in ms']
dlg2 = gui.DlgFromDict(dictionary=Info, title='Experiment Info', order=dialog_order)

if dlg2.OK:
    on_ms = int(Info['time for stimuli in ms'])
    numTrial = int(Info['Number for 0 lag'])
    duration_ms = int(Info['Max duration in ms'])
else:
    core.quit()

on_s = on_ms / 1000           # convert to seconds (per your template)
duration = duration_ms / 1000
ITI_s = (1, 1.5)

today = date.today()
today_str = today.strftime("%m%d%Y")
lagList = []
DEFAULT_LAGLIST = [-0.5, 0, 0.5]
# create folder and update record
directory_path = Path(r"Z:\16-Mouse-HRD\1-data\1-raw\Subjects")
today = date.today()
today_str = today.strftime("%m%d%Y")
DEFAULT_LAGLIST = [-0.5, 0, 0.5]

mouse_path = directory_path / mouseName
mouse_path.mkdir(parents=True, exist_ok=True)
record_path = mouse_path / 'DataRecord.csv'

# defaults
session_already = 0
trial_already = 0
lagList = list(DEFAULT_LAGLIST)
ifTrial_200 = False

# read last row of record file (if exists) to get session/trial/lag info
if record_path.exists():
    try:
        with open(record_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            last_row = None
            for row in reader:
                if row:
                    last_row = row
            if last_row:
                try:
                    session_already = int(last_row[0])
                except Exception:
                    session_already = 0
                try:
                    trial_already = int(last_row[4])
                except Exception:
                    trial_already = 0

                try:
                    parsed = ast.literal_eval(last_row[-2])
                    if isinstance(parsed, (list, tuple)) and parsed:
                        lagList = [float(x) for x in parsed]
                except Exception as e:
                    print(e)
    except Exception as e:
        print(f"Warning: failed reading {record_path}: {e}")
else:
    with open(record_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Session", "Date", "LagList for this experiment", "Performance", "Total Trial", "LagList for next experiment", "ifAnalyzed"])  # header row

# create new session folder using a string foldername
nextSession = session_already + 1
foldername = f"Session {nextSession} - {today_str}"
full_path = mouse_path / foldername
full_path.mkdir(parents=True, exist_ok=True)

# set wheelGain based on trial_already
ifTrial_200 = trial_already > 200
wheelGain = 4 if ifTrial_200 else 8

print("wheelGain:", wheelGain)
print("lagList:", lagList)

dataStorage = []

# audio stimuli
sample_rate = 48000
CS = 0.5   # audio baseline (0..1)

# visual stimuli
frames_per_sec = 60
CL = -0.2     # visual baseline (-1..1)

#Trial Number
counts = [0] * len(lagList)
middle = len(lagList) // 2
counts[middle] = int(numTrial)

for i in range(len(counts)):
    if i != middle:
        counts[i] = math.ceil(numTrial / (len(lagList) - 1))

print(counts)

selection = []
for value, count in zip(lagList, counts):
    selection.extend([value] * count)
random.shuffle(selection)

# recompute trial_numbers/reaction_times to match selection length
trial_numbers = list(range(1, len(selection) + 1))
reaction_times = [0] * len(selection)

# for probability normalization later (ensure counts are ints)
counts_dict = {lagval: int(cnt) for lagval, cnt in zip(lagList, counts)}

# session noice
total_ITI_estimate = sum([random.uniform(*ITI_s) for _ in selection])
total_session_duration = len(selection) * duration + total_ITI_estimate
session_samples = generate_pink_noise(2*total_session_duration, sample_rate=sample_rate)
stereo_session = np.column_stack([session_samples, session_samples])
sessionSound = sound.Sound(value=stereo_session, sampleRate=sample_rate, stereo=True)
sessionSound.play()
core.wait(0.01) 

#Continuous and discountinuous data record setup
newFile_con = os.path.join(full_path, "continuous.csv")
field = ["frame","trial", "time", "x position", "ITI"]
with open(newFile_con, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(field)

newFile_dis = os.path.join(full_path, "discontinuous.csv")
field = ["trial", "on_s", "Time Lag", "Reaction time", "choice", "reward amount"]
with open(newFile_dis, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(field)

globalFrames = 0
globalTime = core.Clock()
left_choice_amount = 0
right_choice_amount = 0
no_choice_amount = 0

#plot data storage
left_storage = [0] * len(lagList)
left_prob = [0] * len(lagList)
total_0 = numTrial
total_other = math.ceil(numTrial / (len(lagList) - 1))

#plot
plt.ion()
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
line1, = ax[0].plot(lagList, left_prob, 'bo-', label='Left Choice')
ax[0].set_xlabel("Lag (s)")
ax[0].set_ylabel("Left Choice Probability")
ax[0].set_ylim(0, 1)
ax[0].set_xticks(lagList)
ax[0].set_title("Left Choice Probability vs Time Lag")

trial_numbers = list(range(1, len(selection) + 1))
reaction_times = [0] * len(selection)
line2, = ax[1].plot(trial_numbers, reaction_times, 'go-')
ax[1].set_xlabel("Trial Number")
ax[1].set_ylabel("Reaction Time (s)")
ax[1].set_title("Reaction Time vs Trial Number")

total_reward = 0
bars = ax[2].bar([0], [total_reward], color='purple')
ax[2].set_xticks([])
ax[2].set_ylabel("Cumulative Reward")
ax[2].set_title("Reward Accumulation")
ax[2].set_ylim(0, numTrial * len(lagList))

# --- Main experiment loop ---
# Pre-trial ITI at session start (optional small segment already in session_samples)
session_pointer = 0
ITI_start = random.uniform(*ITI_s)
n_samples_ITI_start = int(ITI_start * sample_rate)
if n_samples_ITI_start > 0:
    iti_slice = session_samples[session_pointer : session_pointer + n_samples_ITI_start] * CS
    stereo_iti = np.column_stack([iti_slice, iti_slice])
    itiSound = sound.Sound(value=stereo_iti, sampleRate=sample_rate, stereo=True)
    itiSound.play()
    circle.fillColor = [CL]*3
    circle.pos = [0,0]
    circle.draw()
    win.flip()
    core.wait(ITI_start)
    itiSound.stop()
    session_pointer += n_samples_ITI_start

#Amplitude
AS = 0.3
AV = 0.7
pattern_S = [+AS, -AS] 
pattern_V = [+AV, -AV]
pattern_lenS = len(pattern_S)
pattern_lenV = len(pattern_V)
TS = on_s                     
frames_per_sec = 60
actual_trial = 0

callForStop = False
trialWhenBreak = 0
kb.clock.reset()  

# ---- Run trials ----
for k, lag in enumerate(selection):
    print(f"Trial {k+1}, lag={lag}")
    touchTime = -1
    choice = "none"
    frozen = False
    ITI = random.uniform(*ITI_s)

    # Reset encoder and circle
    m.set_zero_position()
    core.wait(0.05)
    circlePos = [0, 0]
    reference_pos = m.current_position()
    trial_clock = core.Clock()

    totalFrames = int(duration * frames_per_sec)
    
    for i in range(totalFrames):
        globalFrames += 1
        t = trial_clock.getTime()

        elapsed_visual = max(0, t)        # visual starts immediately
        elapsed_audio  = max(0, t - lag)  # audio starts after lag

        step_index_visual = int(elapsed_visual // TS)
        step_index_audio  = int(elapsed_audio // TS)

        if elapsed_visual >= 0:
            val_visual = pattern_V[step_index_visual % pattern_lenV]
        else:
            val_visual = 0

        if elapsed_audio >= 0:
            val_audio = pattern_S[step_index_audio % pattern_lenS]
        else:
            val_audio = CS  

        # --- Update audio & visual ---
        sessionSound.setVolume(CS + val_audio)
        circle.fillColor = [val_visual, val_visual, val_visual]
        circle.pos = circlePos
        circle.draw()
        win.flip()

        # --- Update encoder / circle position ---
        if not frozen:
            current_pos = m.current_position()
            delta = current_pos - reference_pos
            circlePos[0] += delta * wheelGain
            reference_pos = current_pos

       # non-blocking getKeys each frame; look for 'escape'
        keys = kb.getKeys(keyList=['escape'], waitRelease=False)
        if any(k.name == 'escape' for k in keys):
            callForStop = True
            trialWhenBreak = k + 1
            break

        #Edge detection
        left_edge = -length/2 + radius
        right_edge = length/2 - radius
        if not frozen:
            if circlePos[0] <= left_edge:
                touchTime = trial_clock.getTime()
                choice = "left"
                left_choice_amount += 1
                frozen = True
                # increment left storage for matching lag index
                for idx_l in range(len(lagList)):
                    if lagList[idx_l] == lag:
                        left_storage[idx_l] += 1
                for idx_lp in range(len(left_prob)):
                    denom = counts_dict[lagList[idx_lp]]
                    left_prob[idx_lp] = left_storage[idx_lp] / denom
                reaction_times[k] = touchTime
                line1.set_ydata(left_prob)
                line1.set_xdata(lagList)
                line2.set_ydata(reaction_times)
                line2.set_xdata(trial_numbers)
                if lag != 0:
                    total_reward += 1
                    bars[0].set_height(total_reward)
                for a in ax:
                    a.relim(); a.autoscale_view()
                fig.canvas.draw(); fig.canvas.flush_events()
                break

            elif circlePos[0] >= right_edge:
                touchTime = trial_clock.getTime()
                choice = "right"
                right_choice_amount += 1
                frozen = True
                reaction_times[k] = touchTime
                line2.set_ydata(reaction_times)
                line2.set_xdata(trial_numbers)
                if lag == 0:
                    total_reward += 1
                    bars[0].set_height(total_reward)
                for a in ax:
                    a.relim(); a.autoscale_view()
                fig.canvas.draw(); fig.canvas.flush_events()
                break
        
        # Trial timeout
        if trial_clock.getTime() > duration and choice == "none":
            no_choice_amount += 1
            reaction_times[k] = touchTime
            line2.set_ydata(reaction_times)
            line2.set_xdata(trial_numbers)
            for a in ax:
                a.relim(); a.autoscale_view()
            fig.canvas.draw(); fig.canvas.flush_events()
            break

        # Discontinuous data storage
        dataStorage.append([globalFrames, k+1, globalTime.getTime(), circlePos[0],
                            ITI, on_s, lag, touchTime, choice, total_reward])
        

    # Saving file
    print("storing data")
    newData = [globalFrames, (k+1), globalTime.getTime(), circlePos[0], ITI, on_s, lag, touchTime, choice, total_reward]
    try:
        with open(newFile_con, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for r in dataStorage:
                writer.writerow(r)
            writer.writerow(newData)
            print("saved")
    except FileNotFoundError:
        print(f"Error: The file '{newFile_con}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # clear frame-level store for next trial
    dataStorage = []

    # --- Post-trial ITI ---
    core.wait(20/1000) #reward time period
    circlePos = [0,0]
    circle.pos = circlePos
    circle.fillColor = [CL]*3
    circle.draw()
    win.flip()
    # Baseline audio during ITI
    sessionSound.setVolume(CS)

    # Track encoder movement, only start new trial when stop moving wheel
    iti_clock = core.Clock()
    last_move_time = iti_clock.getTime()
    reference_pos = m.current_position()

    while True:
        current_pos = m.current_position()
        delta = current_pos - reference_pos
        reference_pos = current_pos

        if abs(delta) > 0:
            last_move_time = iti_clock.getTime()

        # End ITI if no movement for 1 second
        if iti_clock.getTime() - last_move_time >= 1.0:
            break

    print("storing data")
    actual_trial = k+1
    #store data to output file
    newData = [globalFrames, (k+1), globalTime.getTime(), circlePos[0], ITI]
    try:
        with open(newFile_con, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for q in range(len(dataStorage)):
                writer.writerow(dataStorage[q])
            print("saved")
            writer.writerow(newData)
    except FileNotFoundError:
        print(f"Error: The file '{newFile_con}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    newData = [(k+1), on_s, lag, touchTime, choice, total_reward]
    try:
        with open(newFile_dis, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for q in range(len(dataStorage)):
                writer.writerow(dataStorage[q])
            print("saved")
            writer.writerow(newData)
    except FileNotFoundError:
        print(f"Error: The file '{newFile_dis}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    if callForStop:
        break

    #check if the session should end based on cumulateive behaviour
    if touchTime != -1:
        totalRTNum += touchTime
        totalRTArray.append(touchTime)
        if((k+1) % trialNum != 0):
            certainRT.append(touchTime)
        else:
            certainRT = [touchTime]
    else:
        totalRTNum += duration
        totalRTArray.append(duration)
        if((k+1) % trialNum != 0):
            certainRT.append(duration)
        else:
           certainRT = [duration]
    medianRT = statistics.median(totalRTArray)
    medianCertainRT = statistics.median(certainRT)
    print(totalRTNum)
    print(k+1)
    print(medianRT)
    print(certainRT)
    result = session_ending(totalRTNum, k + 1, medianRT, medianCertainRT)
    if result == True:
        print("should end")
        break

plot_path = full_path / "behavior_summary.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {plot_path}")

#Final update for data record file
trial_already += actual_trial
if callForStop == False or (callForStop == True and trialWhenBreak > 30):
    # Add new row with actual session info
    if actual_trial < 2 * numTrial:
        # If the behavior ends because of behavior is not ideal, update immediately as bad performance
        newRow = [session_already + 1, today_str, lagList, 'bad', trial_already, lagList, True]
    else:
        data_path = full_path / "discontinuous.csv"
        time_lag = []
        lag_to_choice = []
        probability = []
        performance = ""

        try:
            with open(data_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header_data = next(reader)  # skip header
                for row in reader:
                    lag_value = float(row[2].strip())
                    lag_to_choice.append([lag_value, row[4]])
                    if lag_value not in time_lag:
                        time_lag.append(lag_value)

            time_lag = sorted(set(time_lag))

            # Find index closest to 0.0
            if time_lag:
                position = min(range(len(time_lag)), key=lambda i: abs(time_lag[i] - 0.0))
            else:
                position = None

            # Calculate probability per lag 
            for lag in time_lag:
                count = 0
                left_count = 0
                for l, choice in lag_to_choice:
                    if abs(lag - l) < 1e-6:  # float tolerance
                        count += 1
                        if choice.strip().lower() == "left":
                            left_count += 1
                prob = left_count / count if count > 0 else 0
                probability.append(prob)
                print(f"lag={lag}, count={count}, left_count={left_count}, prob={prob}")

            print("Probability per lag:", probability)

            # Determine performance
            if position is None or len(probability) < 2:
                determine = 0
            else:
                pos_minus = max(0, position-1)
                pos_plus = min(len(probability)-1, position+1)
                determine = (probability[pos_minus] + probability[pos_plus])/2 - probability[position]

            print(f"Determine value: {determine}")

            performance = "good" if determine > 0.7 else "bad"

            if performance == "good" and position is not None:
                nextLag = [None] * (len(time_lag) + 2)
                for k in range(len(nextLag)//2 + 1):
                    nextLag[k] = float(time_lag[k])
                    nextLag[-(k+1)] = float(time_lag[-(k+1)])
                if position - 1 >= 0:
                    nextLag[position] = float(time_lag[position - 1]) + 0.1
                else:
                    nextLag[position] = float(time_lag[position])
                if position + 1 < len(nextLag):
                    nextLag[position + 1] = 0.0
                if position + 2 < len(nextLag) and position + 1 < len(time_lag):
                    nextLag[position + 2] = float(time_lag[position + 1]) - 0.1
            else:
                nextLag = time_lag.copy()

        except FileNotFoundError:
            print(f"Data file not found: {data_path}")
            nextLag = list(lagList)

        newRow = [session_already + 1, today_str, lagList, performance, trial_already, nextLag, True]
else:
    print("quit")
    core.quit()

try:
    with open(record_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(newRow)
except Exception as e:
    print(f"Failed to append to {record_path}: {e}")


sessionSound.stop()
win.close()
left_storage
