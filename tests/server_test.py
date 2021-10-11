import json
import time

init = {'1': True, '2': False}

with open('test.json', 'w') as write_file:
    json.dump(init, write_file)

for i in range(10):
    print(i)
    time.sleep(4)

init['2'] = True

with open('test.json', 'w') as write_file:
    json.dump(init, write_file)