import matplotlib.pyplot as plt
import string
import re
import math

files=[]

files.append('histograms/train.question.histogram')
files.append('histograms/train.context.histogram')
files.append('histograms/train.answer.histogram')

files.append('histograms/val.question.histogram')
files.append('histograms/val.context.histogram')
files.append('histograms/val.answer.histogram')

files.append('histograms/dev.question.histogram')
files.append('histograms/dev.context.histogram')
files.append('histograms/dev.answer.histogram')

fig, f = plt.subplots()
for i in range(3):
    for j in range(3):
        x = []
        y = []
        with open(files[3*i+j]) as cfiles:
            for line in cfiles:
                if len(line) > 0:
                    t = [int(s) for s in re.findall(r'\b\d+\b',line)]
                    x.append(float(t[0]))
                    y.append(math.log(t[1],2.))
        zipped = zip(x,y)
        zz = sorted(zipped, key=lambda tt:tt[0])
        x,y=zip(*zz)
        f.plot(x,y,label=files[3*i+j])
        cfiles.close()
        print files[3*i+j]
plt.legend(loc='upper right', shadow=True)
plt.title('All histograms piled up')
plt.ylabel('Length counts (base 2)')
plt.xlabel('Sentence/Paragraph lengths in words')
plt.show()
