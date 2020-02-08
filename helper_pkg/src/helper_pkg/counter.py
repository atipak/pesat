from itertools import count

types = set()
count = 0
types.add("10/random/0.4/nothing/-1/yes/1")
types.add("100/random/0.4/far/-1/yes/1")
for size in ["10", "100"]:
    for setting in ["random", "grid"]:
        for bu in ["0.1", "0.4"]:
            for position in ["near", "far"]:
                for height in ["-1", "1.5"]:
                    types.add("{}/{}/{}/{}/{}/no/2".format(size, setting, bu, position, height))
                    count += 1

for bu in ["0.1", "0.4", "0.25"]:
    types.add("100/random/{}/near/-1/no/2".format(size, setting, bu, position, height))
    count += 1

for size in ["10", "100", "1000"]:
    for setting in ["random"]:
        for bu in ["0.25"]:
            for position in ["near"]:
                for height in ["-1", "1.5", "20"]:
                    types.add("{}/{}/{}/{}/{}/no/2".format(size, setting, bu, position, height))
                    count += 1

for size in ["100"]:
    for setting in ["random", "grid"]:
        for bu in ["0.25"]:
            for position in ["far"]:
                for height in ["-1"]:
                    types.add("{}/{}/{}/{}/{}/no/2".format(size, setting, bu, position, height))
                    count += 1

for size in ["100", "1000"]:
    for setting in ["random"]:
        for bu in ["0.25", "0.1", "0.4"]:
            for position in ["far"]:
                for height in ["-1"]:
                    types.add("{}/{}/{}/{}/{}/no/3".format(size, setting, bu, position, height))
                    count += 1

import copy
types = list(types)
sorted_types = []
already_added = []
for i in range(len(types)):
    typ = types[i].split("/")
    count = typ[6]
    substr = list(typ[:6])
    if substr not in already_added:
        for j in range(i + 1, len(types)):
            typ2  = types[j].split("/")
            count2 = typ2[6]
            substr2 = typ2[:6]
            if set(substr) == set(substr2):
                if count < count2:
                    count = count2
        substr.append(count)
        sorted_types.append(substr)
        already_added.append(substr)


for line in sorted_types:
    print(line)

print(len(types))
print(len(sorted_types))
print(count)
