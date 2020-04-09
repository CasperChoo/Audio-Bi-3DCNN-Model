import os
import subprocess
import os
import zipfile
training_set_data_path="/content/drive/My Drive/Deep Learning/CAER/train(new)"
entries=os.listdir(training_set_data_path)
for entry in entries:
        entries2=os.listdir(training_set_data_path+"/"+entry+'/')
        for file_name in entries2:
          avi_file_path=''
          output_name=''
          file_name=(file_name.split('.avi'))[0]
          try:
              if not os.path.exists(f'train(mp4)/{entry}/'):
                  os.makedirs(f'train(mp4)/{entry}/')
          except OSError:
              print ('Error: Creating directory of data')
          avi_file_path=training_set_data_path+"/"+str(entry)+'/'str(file_name)+'.avi'
          output_name=f'train(mp4)/{entry}/{file_name}'
          os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 'train(mp4)/{output}.mp4'".format(input = avi_file_path, output = output_name))
       
entries=os.listdir('train(mp4)/')
for entry in entries:
        entries2=os.listdir('train(mp4)/'+entry+'/')
        for file_name in entries2:
          file_name=(file_name.split('.mp4'))[0]
          try:
              if not os.path.exists(f'VoiceData/{entry}/'):
                  os.makedirs(f'VoiceData/{entry}}/')
          except OSError:
              print ('Error: Creating directory of data')
          command = "ffmpeg -i train(mp4)/{}/{}.mp4 -ab 320k -ac 2 -ar 44100 -vn VoiceData/{}/{}.wav".format(entry,file_name,entry,file_name)    
          subprocess.call(command, shell=True)
