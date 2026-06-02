from binarytree import Node, tree

learnedTrees = """Tree 0:
small | 
yellow | nestspot | 
fast | newworld | lean | red | 
rhinocerus | deer | lion | grizzly+bear | skunk | weasel | raccoon | fox | 
---------
Tree 1:
smelly | 
white | white | 
horns | flippers | meatteeth | meat | 
chihuauah | deer | siamese+cat | seal | antelope | chihuauah | cow | rat | 
---------
Tree 2:
paws | 
swims | forest | 
muscle | walks | inactive | bush | 
pig | deer | humpbac+whale | hippopotamus | rat | persian | raccoon | tiger | 
---------
Tree 3:
plains | 
forest | arctic | 
patches | hands | weak | hairless | 
seal | dalmatian | hamster | spider+monkey | deer | sheep | moose | ox | 
---------
Tree 4:
claws | 
meat | chewteeth | 
paws | forest | gray | tunnels | 
antelope | rabbit | collie | rabbit | leopard | rat | bat | hamster | 
---------
Tree 5:
fields | 
bush | mountains | 
toughskin | furry | big | longleg | 
dalmatian | dolphin | elephant | tiger | cow | cow | sheep | deer | 
---------
Tree 6:
forager | 
hibernate | gray | 
paws | hibernate | tail | toughskin | 
giraffe | collie | humpbac+whale | beaver | giant+panda | chimpan | rat | bat | 
---------
Tree 7:
nestspot | 
toughskin | tree | 
forest | buckteeth | white | fast | 
blue+whale | raccoon | cow | horse | beaver | hamster | giant+panda | raccoon | 
---------
Tree 8:
timid | 
big | patches | 
yellow | paws | hunter | hunter | 
mole | bobcat | gorilla | leopard | ox | seal | cow | collie | 
---------
Tree 9:
swims | 
horns | chewteeth | 
active | tusks | ocean | quadrapedal | 
pig | dalmatian | moose | rhinocerus | humpbac+whale | seal | dolphin | otter | 
---------
Tree 10:
lean | 
nocturnal | jungle | 
horns | forest | pads | brown | 
hippopotamus | cow | hamster | grizzly+bear | giraffe | wolf | siamese+cat | bat | 
---------
Tree 11:
hibernate | 
small | vegetation | 
patches | water | big | mountains | 
rhinocerus | pig | persian | seal | bat | grizzly+bear | hamster | bat | 
---------
Tree 12:
solitary | 
pads | longleg | 
furry | plains | agility | vegetation | 
humpbac+whale | sheep | lion | german+shepherd | grizzly+bear | polar+bear | german+shepherd | moose | 
---------
Tree 13:
group | 
spots | ground | 
stalker | forager | plankton | tail | 
collie | fox | seal | bobcat | seal | humpbac+whale | rhinocerus | deer | 
---------
Tree 14:
swims | 
fields | patches | 
agility | small | hunter | furry | 
grizzly+bear | lion | cow | hamster | dolphin | seal | killer+whale | otter | 
---------
Tree 15:
meat | 
newworld | claws | 
strainteeth | smelly | bulbous | forest | 
gorilla | hippopotamus | deer | cow | collie | cow | raccoon | raccoon | 
---------
Tree 16:
tunnels | 
tree | mountains | 
newworld | flys | hands | paws | 
zebra | siamese+cat | bobcat | bat | hamster | spider+monkey | bat | bat | 
---------
Tree 17:
slow | 
hooves | group | 
quadrapedal | tunnels | fast | smart | 
bat | hamster | deer | deer | ox | polar+bear | buffalo | seal | 
---------
Tree 18:
smelly | 
swims | gray | 
paws | jungle | forest | strong | 
zebra | squirre | dolphin | hippopotamus | cow | chimpan | hamster | horse | 
---------
Tree 19:
forest | 
paws | ocean | 
gray | plains | small | ground | 
sheep | rhinocerus | dalmatian | german+shepherd | moose | rat | otter | mouse | 
---------
Tree 20:
quadrapedal | 
forest | claws | 
water | walks | mountains | coastal | 
cow | humpbac+whale | bat | deer | cow | deer | wolf | polar+bear | 
---------
Tree 21:
strong | 
domestic | hands | 
buckteeth | fish | bulbous | domestic | 
squirre | squirre | collie | siamese+cat | horse | pig | gorilla | horse | 
---------
Tree 22:
fish | 
meat | paws | 
paws | weak | forager | strong | 
deer | hamster | chihuauah | rat | dolphin | humpbac+whale | otter | grizzly+bear | 
---------
Tree 23:
ocean | 
claws | hairless | 
grazer | fierce | weak | coastal | 
gorilla | rabbit | weasel | raccoon | otter | otter | humpbac+whale | dolphin | 
---------
Tree 24:
brown | 
hooves | buckteeth | 
tail | stripes | paws | tunnels | 
giant+panda | persian | sheep | zebra | cow | collie | horse | hamster | 
---------
Tree 25:
forest | 
furry | horns | 
flippers | tail | coastal | furry | 
giraffe | seal | buffalo | cow | chihuauah | otter | moose | deer | 
---------
Tree 26:
fierce | 
bush | nestspot | 
vegetation | big | fields | cave | 
humpbac+whale | moose | hamster | giraffe | polar+bear | german+shepherd | raccoon | grizzly+bear | 
---------
Tree 27:
furry | 
forager | strong | 
bush | gray | nestspot | solitary | 
seal | rhinocerus | antelope | walrus | chihuauah | mouse | spider+monkey | ox | 
---------
Tree 28:
longleg | 
group | plains | 
meatteeth | paws | black | newworld | 
hamster | raccoon | sheep | tiger | elephant | gorilla | zebra | leopard | 
---------
Tree 29:
small | 
smelly | claws | 
hairless | pads | water | meatteeth | 
antelope | hippopotamus | ox | polar+bear | gorilla | dolphin | squirre | raccoon | 
---------
Tree 30:
hooves | 
small | lean | 
cave | mountains | domestic | ocean | 
dolphin | wolf | hamster | bat | moose | cow | antelope | moose | 
---------
Tree 31:
slow | 
forager | smelly | 
small | desert | fields | strong | 
killer+whale | collie | deer | lion | grizzly+bear | hippopotamus | sheep | cow | 
---------
Tree 32:
small | 
jungle | weak | 
longneck | grazer | longleg | grazer | 
german+shepherd | horse | humpbac+whale | elephant | weasel | chimpan | chihuauah | rabbit | 
---------
Tree 33:
scavenger | 
nestspot | hibernate | 
horns | hands | domestic | stripes | 
german+shepherd | cow | otter | chimpan | wolf | wolf | bat | raccoon | 
---------
Tree 34:
arctic | 
patches | group | 
stalker | nocturnal | stalker | ocean | 
moose | grizzly+bear | horse | hamster | otter | polar+bear | moose | humpbac+whale | 
---------
Tree 35:
chewteeth | 
tusks | paws | 
small | hooves | tusks | blue | 
humpbac+whale | seal | leopard | rhinocerus | hippopotamus | elephant | otter | persian | 
---------
Tree 36:
inactive | 
longleg | jungle | 
fields | jungle | flys | tree | 
beaver | horse | german+shepherd | chimpan | sheep | bat | cow | giant+panda | 
---------
Tree 37:
nocturnal | 
plains | plains | 
muscle | nestspot | fierce | newworld | 
chihuauah | bat | sheep | deer | hamster | raccoon | german+shepherd | rat | 
---------
Tree 38:
patches | 
oldworld | smart | 
orange | hairless | walks | forager | 
walrus | fox | fox | hippopotamus | rabbit | cow | seal | raccoon | 
---------
Tree 39:
plains | 
forager | smart | 
weak | hops | domestic | muscle | 
dolphin | siamese+cat | raccoon | squirre | giraffe | cow | deer | deer | 
---------
Tree 40:
group | 
muscle | claws | 
fish | meatteeth | longneck | desert | 
squirre | raccoon | ox | grizzly+bear | chimpan | moose | bat | lion | 
---------
Tree 41:
small | 
brown | hibernate | 
white | bulbous | longneck | timid | 
antelope | tiger | horse | cow | chihuauah | collie | rat | hamster | 
---------
Tree 42:
plains | 
weak | mountains | 
muscle | slow | smart | forest | 
walrus | humpbac+whale | bat | blue+whale | giraffe | german+shepherd | sheep | moose | 
---------
Tree 43:
fish | 
bulbous | fish | 
meatteeth | nocturnal | toughskin | pads | 
horse | weasel | cow | rat | otter | bat | seal | siamese+cat | 
---------
Tree 44:
grazer | 
forest | tusks | 
active | cave | agility | coastal | 
persian | siamese+cat | chimpan | mouse | ox | hamster | rhinocerus | walrus | 
---------
Tree 45:
smelly | 
fish | slow | 
bulbous | walks | white | black | 
antelope | moose | dolphin | persian | chimpan | pig | moose | pig | 
---------
Tree 46:
stalker | 
nestspot | toughskin | 
longneck | walks | domestic | brown | 
cow | zebra | walrus | spider+monkey | bobcat | chihuauah | polar+bear | grizzly+bear | 
---------
Tree 47:
claws | 
furry | small | 
smelly | meatteeth | stalker | yellow | 
hippopotamus | walrus | moose | chimpan | beaver | wolf | otter | bobcat | 
---------
Tree 48:
gray | 
fast | fierce | 
bipedal | grazer | water | hooves | 
cow | giant+panda | fox | deer | hamster | humpbac+whale | bat | pig | 
---------
Tree 49:
small | 
nocturnal | fields | 
fierce | spots | water | domestic | 
humpbac whale | buffalo | tiger | leopard | collie | beaver | rat | hamster | 
"""

learnedTrees = learnedTrees.split("---------\n")
setje = set()

for treeStr in learnedTrees:
    break
    lines = treeStr.splitlines()
    print(lines[0])
    root = Node(lines[1].split(" | ")[0])
    root.left = Node(lines[2].split(" | ")[0])
    root.right = Node(lines[2].split(" | ")[1])
    root.left.left = Node(lines[3].split(" | ")[0])
    root.left.right = Node(lines[3].split(" | ")[1])
    root.right.left = Node(lines[3].split(" | ")[2])
    root.right.right = Node(lines[3].split(" | ")[3])
    root.left.left.left = Node(lines[4].split(" | ")[0])
    root.left.left.right = Node(lines[4].split(" | ")[1])
    root.left.right.left = Node(lines[4].split(" | ")[2])
    root.left.right.right = Node(lines[4].split(" | ")[3])
    root.right.left.left = Node(lines[4].split(" | ")[4])
    root.right.left.right = Node(lines[4].split(" | ")[5])
    root.right.right.left = Node(lines[4].split(" | ")[6])
    root.right.right.right = Node(lines[4].split(" | ")[7])
    print(root)

    t1 = (root.value, root.right.value)
    t2 = (root.value, root.right.left.value)
    t3 = (root.left.value, root.left.right.value)
    setje.add(t1)
    setje.add(t2)
    setje.add(t3)
#raise Exception("stop")
#for item in setje:
#    print(item)

learnedTrees = """Tree 0:
R0 | 
L4 | L7 | 
R5 | R2 | L2 | L4 | 
7 | 8 | 6 | 6 | 4 | 4 | 9 | 4 | 
---------
Tree 1:
R1 | 
L6 | R1 | 
R3 | L5 | R6 | L2 | 
13 | 11 | 6 | 6 | 7 | 6 | 8 | 3 | 
---------
Tree 2:
L5 | 
L0 | L5 | 
R9 | R8 | L0 | L5 | 
7 | 12 | 0 | 8 | 7 | 8 | 14 | 14 | 
---------
Tree 3:
L3 | 
R4 | R1 | 
L1 | L3 | R9 | R0 | 
9 | 10 | 6 | 4 | 3 | 10 | 4 | 3 | 
---------
Tree 4:
R7 | 
L1 | R0 | 
L5 | L6 | R0 | L6 | 
15 | 13 | 1 | 15 | 13 | 7 | 1 | 13 | 
---------
Tree 5:
L2 | 
R8 | R0 | 
L0 | R4 | L4 | L0 | 
10 | 3 | 10 | 9 | 3 | 4 | 2 | 3 | 
---------
Tree 6:
L8 | 
L3 | R6 | 
R8 | R4 | L2 | R2 | 
12 | 13 | 7 | 7 | 4 | 5 | 8 | 10 | 
---------
Tree 7:
L8 | 
L4 | L4 | 
R2 | R0 | R1 | L7 | 
2 | 8 | 8 | 8 | 9 | 9 | 9 | 10 | 
---------
Tree 8:
L5 | 
L4 | L4 | 
L4 | R3 | L4 | L4 | 
8 | 4 | 4 | 14 | 5 | 5 | 5 | 4 | 
---------
Tree 9:
R4 | 
L0 | R2 | 
L1 | L7 | L7 | L0 | 
14 | 2 | 2 | 2 | 12 | 14 | 17 | 2 | 
---------
Tree 10:
L3 | 
R0 | L7 | 
L1 | L1 | R8 | L1 | 
14 | 3 | 8 | 3 | 5 | 14 | 5 | 5 | 
---------
Tree 11:
L7 | 
R7 | L1 | 
L7 | L6 | L6 | R7 | 
12 | 11 | 11 | 13 | 9 | 13 | 9 | 13 | 
---------
Tree 12:
R8 | 
R2 | R7 | 
R7 | L8 | R1 | R2 | 
6 | 9 | 9 | 10 | 9 | 6 | 9 | 7 | 
---------
Tree 13:
R1 | 
L7 | L6 | 
R2 | R5 | R3 | R1 | 
14 | 7 | 13 | 14 | 9 | 9 | 9 | 9 | 
---------
Tree 14:
L1 | 
R4 | R3 | 
L2 | R9 | R1 | R7 | 
8 | 5 | 8 | 8 | 8 | 8 | 10 | 8 | 
---------
Tree 15:
L9 | 
R5 | R1 | 
R0 | L4 | L9 | L5 | 
13 | 7 | 5 | 7 | 12 | 17 | 18 | 17 | 
---------
Tree 16:
R3 | 
L6 | R4 | 
R0 | L6 | L6 | R1 | 
8 | 2 | 9 | 9 | 9 | 9 | 9 | 10 | 
---------
Tree 17:
R4 | 
L4 | L8 | 
L2 | R9 | L0 | L8 | 
8 | 5 | 11 | 13 | 11 | 4 | 4 | 8 | 
---------
Tree 18:
R4 | 
L2 | R2 | 
R0 | R0 | R2 | L6 | 
11 | 8 | 2 | 2 | 10 | 2 | 2 | 7 | 
---------
Tree 19:
L5 | 
R8 | L1 | 
R9 | L0 | R7 | R5 | 
6 | 17 | 8 | 8 | 11 | 12 | 12 | 6 | 
---------
Tree 20:
R3 | 
R1 | R7 | 
L9 | R5 | L1 | L5 | 
9 | 16 | 1 | 7 | 8 | 4 | 16 | 11 | 
---------
Tree 21:
L7 | 
R1 | L1 | 
L1 | L1 | R0 | L7 | 
10 | 4 | 9 | 2 | 14 | 7 | 2 | 7 | 
---------
Tree 22:
R3 | 
L6 | L5 | 
L5 | L5 | L6 | R3 | 
10 | 13 | 7 | 5 | 6 | 7 | 14 | 6 | 
---------
Tree 23:
L4 | 
R7 | L4 | 
L0 | L0 | L6 | L4 | 
12 | 3 | 8 | 7 | 7 | 9 | 3 | 9 | 
---------
Tree 24:
L1 | 
R9 | R0 | 
R4 | L1 | L9 | L1 | 
14 | 13 | 14 | 14 | 7 | 14 | 13 | 7 | 
---------
Tree 25:
R7 | 
L2 | L2 | 
R7 | R0 | L9 | R5 | 
16 | 11 | 7 | 12 | 11 | 13 | 4 | 7 | 
---------
Tree 26:
L8 | 
L8 | R7 | 
R9 | L8 | R6 | R3 | 
11 | 11 | 15 | 12 | 15 | 6 | 15 | 15 | 
---------
Tree 27:
L0 | 
R9 | L5 | 
L5 | L0 | L2 | L0 | 
10 | 8 | 18 | 9 | 3 | 3 | 3 | 3 | 
---------
Tree 28:
L6 | 
R2 | L6 | 
R1 | R9 | L9 | R0 | 
11 | 4 | 7 | 7 | 6 | 15 | 6 | 6 | 
---------
Tree 29:
L4 | 
R6 | R6 | 
L7 | R3 | L5 | R3 | 
5 | 10 | 9 | 10 | 7 | 5 | 9 | 7 | 
---------
Tree 30:
L0 | 
L4 | L0 | 
R1 | L7 | L3 | R4 | 
10 | 6 | 8 | 9 | 8 | 4 | 7 | 5 | 
---------
Tree 31:
R2 | 
L2 | L1 | 
R3 | R2 | L1 | L4 | 
8 | 12 | 10 | 10 | 9 | 3 | 3 | 3 | 
---------
Tree 32:
L5 | 
L8 | R1 | 
R8 | R6 | R7 | L3 | 
5 | 12 | 17 | 14 | 11 | 12 | 12 | 6 | 
---------
Tree 33:
R9 | 
R6 | R6 | 
R0 | L1 | R9 | R6 | 
7 | 6 | 8 | 8 | 10 | 10 | 10 | 8 | 
---------
Tree 34:
R6 | 
L2 | L0 | 
R2 | L8 | R6 | R5 | 
12 | 11 | 4 | 4 | 12 | 6 | 6 | 6 | 
---------
Tree 35:
R1 | 
L6 | R4 | 
L2 | L0 | L7 | L0 | 
11 | 9 | 12 | 1 | 1 | 11 | 12 | 1 | 
---------
Tree 36:
R9 | 
L4 | R0 | 
L9 | R5 | L8 | R5 | 
13 | 11 | 10 | 9 | 15 | 13 | 15 | 13 | 
---------
Tree 37:
R3 | 
L3 | L1 | 
L9 | R0 | R9 | L5 | 
9 | 10 | 9 | 3 | 8 | 8 | 4 | 4 | 
---------
Tree 38:
R1 | 
R0 | L1 | 
L4 | L9 | L2 | R1 | 
3 | 5 | 6 | 9 | 5 | 3 | 10 | 2 | 
---------
Tree 39:
L4 | 
L1 | L0 | 
R8 | R4 | R6 | R1 | 
15 | 17 | 5 | 5 | 6 | 5 | 12 | 1 | 
---------
Tree 40:
R1 | 
R2 | L3 | 
L3 | L5 | L5 | L3 | 
12 | 11 | 5 | 6 | 8 | 6 | 5 | 11 | 
---------
Tree 41:
R9 | 
L6 | R9 | 
R1 | L6 | L0 | L9 | 
13 | 5 | 14 | 14 | 14 | 8 | 16 | 18 | 
---------
Tree 42:
R4 | 
L1 | L2 | 
R7 | R6 | R7 | L1 | 
15 | 7 | 1 | 7 | 4 | 4 | 6 | 1 | 
---------
Tree 43:
R0 | 
R9 | L5 | 
L2 | L7 | R0 | L0 | 
10 | 8 | 13 | 16 | 12 | 3 | 13 | 13 | 
---------
Tree 44:
R0 | 
R8 | L0 | 
R2 | L9 | L7 | R0 | 
11 | 3 | 9 | 13 | 8 | 8 | 4 | 8 | 
---------
Tree 45:
L3 | 
R1 | R6 | 
R3 | L6 | R3 | L0 | 
13 | 9 | 5 | 9 | 9 | 9 | 9 | 9 | 
---------
Tree 46:
L6 | 
R7 | R7 | 
L2 | L8 | R8 | L8 | 
8 | 10 | 16 | 15 | 12 | 12 | 16 | 15 | 
---------
Tree 47:
L6 | 
L9 | L6 | 
R5 | L3 | R2 | R4 | 
4 | 8 | 10 | 8 | 15 | 3 | 11 | 14 | 
---------
Tree 48:
L6 | 
R0 | L2 | 
L2 | R4 | R7 | L5 | 
9 | 8 | 8 | 9 | 10 | 10 | 8 | 9 | 
---------
Tree 49:
R8 | 
L7 | R8 | 
R5 | L7 | R5 | L6 | 
15 | 5 | 12 | 13 | 9 | 5 | 10 | 14 | 
"""

learnedTrees = learnedTrees.split("---------\n")
setje = set()

for treeStr in learnedTrees:
    break
    lines = treeStr.splitlines()
    print(lines[0])
    root = Node(lines[1].split(" | ")[0])
    root.left = Node(lines[2].split(" | ")[0])
    root.right = Node(lines[2].split(" | ")[1])
    root.left.left = Node(lines[3].split(" | ")[0])
    root.left.right = Node(lines[3].split(" | ")[1])
    root.right.left = Node(lines[3].split(" | ")[2])
    root.right.right = Node(lines[3].split(" | ")[3])
    root.left.left.left = Node(lines[4].split(" | ")[0])
    root.left.left.right = Node(lines[4].split(" | ")[1])
    root.left.right.left = Node(lines[4].split(" | ")[2])
    root.left.right.right = Node(lines[4].split(" | ")[3])
    root.right.left.left = Node(lines[4].split(" | ")[4])
    root.right.left.right = Node(lines[4].split(" | ")[5])
    root.right.right.left = Node(lines[4].split(" | ")[6])
    root.right.right.right = Node(lines[4].split(" | ")[7])
    print(root)

    t1 = (root.value, root.right.value)
    t2 = (root.value, root.right.left.value)
    t3 = (root.left.value, root.left.right.value)
    setje.add(t1)
    setje.add(t2)
    setje.add(t3)


for item in setje:
    print(item)


learnedTrees = """Tree 0:
R0 | 
L4 | L7 | 
R5 | R2 | L2 | L4 | 
7 | 8 | 6 | 6 | 4 | 4 | 9 | 4 | 
---------
Tree 1:
R1 | 
L6 | R1 | 
R3 | L5 | R6 | L2 | 
13 | 11 | 6 | 6 | 7 | 6 | 8 | 3 | 
---------
Tree 2:
L5 | 
L0 | L5 | 
R9 | R8 | L0 | L5 | 
7 | 12 | 0 | 8 | 7 | 8 | 14 | 14 | 
---------
Tree 3:
L3 | 
R4 | R1 | 
L1 | L3 | R9 | R0 | 
9 | 10 | 6 | 4 | 3 | 10 | 4 | 3 | 
---------
Tree 4:
R7 | 
L1 | R0 | 
L5 | L6 | R0 | L6 | 
15 | 13 | 1 | 15 | 13 | 7 | 1 | 13 | 
---------
Tree 5:
L2 | 
R8 | R0 | 
L0 | R4 | L4 | L0 | 
10 | 3 | 10 | 9 | 3 | 4 | 2 | 3 | 
---------
Tree 6:
L8 | 
L3 | R6 | 
R8 | R4 | L2 | R2 | 
12 | 13 | 7 | 7 | 4 | 5 | 8 | 10 | 
---------
Tree 7:
L8 | 
L4 | L4 | 
R2 | R0 | R1 | L7 | 
2 | 8 | 8 | 8 | 9 | 9 | 9 | 10 | 
---------
Tree 8:
L5 | 
L4 | L4 | 
L4 | R3 | L4 | L4 | 
8 | 4 | 4 | 14 | 5 | 5 | 5 | 4 | 
---------
Tree 9:
R4 | 
L0 | R2 | 
L1 | L7 | L7 | L0 | 
14 | 2 | 2 | 2 | 12 | 14 | 17 | 2 | 
---------
Tree 10:
L3 | 
R0 | L7 | 
L1 | L1 | R8 | L1 | 
14 | 3 | 8 | 3 | 5 | 14 | 5 | 5 | 
---------
Tree 11:
L7 | 
R7 | L1 | 
L7 | L6 | L6 | R7 | 
12 | 11 | 11 | 13 | 9 | 13 | 9 | 13 | 
---------
Tree 12:
R8 | 
R2 | R7 | 
R7 | L8 | R1 | R2 | 
6 | 9 | 9 | 10 | 9 | 6 | 9 | 7 | 
---------
Tree 13:
R1 | 
L7 | L6 | 
R2 | R5 | R3 | R1 | 
14 | 7 | 13 | 14 | 9 | 9 | 9 | 9 | 
---------
Tree 14:
L1 | 
R4 | R3 | 
L2 | R9 | R1 | R7 | 
8 | 5 | 8 | 8 | 8 | 8 | 10 | 8 | 
---------
Tree 15:
L9 | 
R5 | R1 | 
R0 | L4 | L9 | L5 | 
13 | 7 | 5 | 7 | 12 | 17 | 18 | 17 | 
---------
Tree 16:
R3 | 
L6 | R4 | 
R0 | L6 | L6 | R1 | 
8 | 2 | 9 | 9 | 9 | 9 | 9 | 10 | 
---------
Tree 17:
R4 | 
L4 | L8 | 
L2 | R9 | L0 | L8 | 
8 | 5 | 11 | 13 | 11 | 4 | 4 | 8 | 
---------
Tree 18:
R4 | 
L2 | R2 | 
R0 | R0 | R2 | L6 | 
11 | 8 | 2 | 2 | 10 | 2 | 2 | 7 | 
---------
Tree 19:
L5 | 
R8 | L1 | 
R9 | L0 | R7 | R5 | 
6 | 17 | 8 | 8 | 11 | 12 | 12 | 6 | 
---------
Tree 20:
R3 | 
R1 | R7 | 
L9 | R5 | L1 | L5 | 
9 | 16 | 1 | 7 | 8 | 4 | 16 | 11 | 
---------
Tree 21:
L7 | 
R1 | L1 | 
L1 | L1 | R0 | L7 | 
10 | 4 | 9 | 2 | 14 | 7 | 2 | 7 | 
---------
Tree 22:
R3 | 
L6 | L5 | 
L5 | L5 | L6 | R3 | 
10 | 13 | 7 | 5 | 6 | 7 | 14 | 6 | 
---------
Tree 23:
L4 | 
R7 | L4 | 
L0 | L0 | L6 | L4 | 
12 | 3 | 8 | 7 | 7 | 9 | 3 | 9 | 
---------
Tree 24:
L1 | 
R9 | R0 | 
R4 | L1 | L9 | L1 | 
14 | 13 | 14 | 14 | 7 | 14 | 13 | 7 | 
---------
Tree 25:
R7 | 
L2 | L2 | 
R7 | R0 | L9 | R5 | 
16 | 11 | 7 | 12 | 11 | 13 | 4 | 7 | 
---------
Tree 26:
L8 | 
L8 | R7 | 
R9 | L8 | R6 | R3 | 
11 | 11 | 15 | 12 | 15 | 6 | 15 | 15 | 
---------
Tree 27:
L0 | 
R9 | L5 | 
L5 | L0 | L2 | L0 | 
10 | 8 | 18 | 9 | 3 | 3 | 3 | 3 | 
---------
Tree 28:
L6 | 
R2 | L6 | 
R1 | R9 | L9 | R0 | 
11 | 4 | 7 | 7 | 6 | 15 | 6 | 6 | 
---------
Tree 29:
L4 | 
R6 | R6 | 
L7 | R3 | L5 | R3 | 
5 | 10 | 9 | 10 | 7 | 5 | 9 | 7 | 
---------
Tree 30:
L0 | 
L4 | L0 | 
R1 | L7 | L3 | R4 | 
10 | 6 | 8 | 9 | 8 | 4 | 7 | 5 | 
---------
Tree 31:
R2 | 
L2 | L1 | 
R3 | R2 | L1 | L4 | 
8 | 12 | 10 | 10 | 9 | 3 | 3 | 3 | 
---------
Tree 32:
L5 | 
L8 | R1 | 
R8 | R6 | R7 | L3 | 
5 | 12 | 17 | 14 | 11 | 12 | 12 | 6 | 
---------
Tree 33:
R9 | 
R6 | R6 | 
R0 | L1 | R9 | R6 | 
7 | 6 | 8 | 8 | 10 | 10 | 10 | 8 | 
---------
Tree 34:
R6 | 
L2 | L0 | 
R2 | L8 | R6 | R5 | 
12 | 11 | 4 | 4 | 12 | 6 | 6 | 6 | 
---------
Tree 35:
R1 | 
L6 | R4 | 
L2 | L0 | L7 | L0 | 
11 | 9 | 12 | 1 | 1 | 11 | 12 | 1 | 
---------
Tree 36:
R9 | 
L4 | R0 | 
L9 | R5 | L8 | R5 | 
13 | 11 | 10 | 9 | 15 | 13 | 15 | 13 | 
---------
Tree 37:
R3 | 
L3 | L1 | 
L9 | R0 | R9 | L5 | 
9 | 10 | 9 | 3 | 8 | 8 | 4 | 4 | 
---------
Tree 38:
R1 | 
R0 | L1 | 
L4 | L9 | L2 | R1 | 
3 | 5 | 6 | 9 | 5 | 3 | 10 | 2 | 
---------
Tree 39:
L4 | 
L1 | L0 | 
R8 | R4 | R6 | R1 | 
15 | 17 | 5 | 5 | 6 | 5 | 12 | 1 | 
---------
Tree 40:
R1 | 
R2 | L3 | 
L3 | L5 | L5 | L3 | 
12 | 11 | 5 | 6 | 8 | 6 | 5 | 11 | 
---------
Tree 41:
R9 | 
L6 | R9 | 
R1 | L6 | L0 | L9 | 
13 | 5 | 14 | 14 | 14 | 8 | 16 | 18 | 
---------
Tree 42:
R4 | 
L1 | L2 | 
R7 | R6 | R7 | L1 | 
15 | 7 | 1 | 7 | 4 | 4 | 6 | 1 | 
---------
Tree 43:
R0 | 
R9 | L5 | 
L2 | L7 | R0 | L0 | 
10 | 8 | 13 | 16 | 12 | 3 | 13 | 13 | 
---------
Tree 44:
R0 | 
R8 | L0 | 
R2 | L9 | L7 | R0 | 
11 | 3 | 9 | 13 | 8 | 8 | 4 | 8 | 
---------
Tree 45:
L3 | 
R1 | R6 | 
R3 | L6 | R3 | L0 | 
13 | 9 | 5 | 9 | 9 | 9 | 9 | 9 | 
---------
Tree 46:
L6 | 
R7 | R7 | 
L2 | L8 | R8 | L8 | 
8 | 10 | 16 | 15 | 12 | 12 | 16 | 15 | 
---------
Tree 47:
L6 | 
L9 | L6 | 
R5 | L3 | R2 | R4 | 
4 | 8 | 10 | 8 | 15 | 3 | 11 | 14 | 
---------
Tree 48:
L6 | 
R0 | L2 | 
L2 | R4 | R7 | L5 | 
9 | 8 | 8 | 9 | 10 | 10 | 8 | 9 | 
---------
Tree 49:
R8 | 
L7 | R8 | 
R5 | L7 | R5 | L6 | 
15 | 5 | 12 | 13 | 9 | 5 | 10 | 14 | 
"""
learnedTrees = learnedTrees.split("---------\n")
setje = set()

counts = {i:0 for i in range(20)}

for treeStr in learnedTrees:
    break
    lines = treeStr.splitlines()
    print(lines[0])
    root = Node(lines[1].split(" | ")[0])
    root.left = Node(lines[2].split(" | ")[0])
    root.right = Node(lines[2].split(" | ")[1])
    root.left.left = Node(lines[3].split(" | ")[0])
    root.left.right = Node(lines[3].split(" | ")[1])
    root.right.left = Node(lines[3].split(" | ")[2])
    root.right.right = Node(lines[3].split(" | ")[3])
    root.left.left.left = Node(lines[4].split(" | ")[0])
    root.left.left.right = Node(lines[4].split(" | ")[1])
    root.left.right.left = Node(lines[4].split(" | ")[2])
    root.left.right.right = Node(lines[4].split(" | ")[3])
    root.right.left.left = Node(lines[4].split(" | ")[4])
    root.right.left.right = Node(lines[4].split(" | ")[5])
    root.right.right.left = Node(lines[4].split(" | ")[6])
    root.right.right.right = Node(lines[4].split(" | ")[7])
    print(root)

    t1 = (root.value, root.right.value)
    t2 = (root.value, root.right.left.value)
    t3 = (root.left.value, root.left.right.value)
    setje.add(t1)
    setje.add(t2)
    setje.add(t3)
    for leaf in lines[4].split(" | "):
        if leaf.isdigit():
            counts[int(leaf)] += 1


for item in setje:
    print(item)


#print("Number of trees with each value as leaf:")
for i in range(20):
    break
    print(f"{i}: {counts[i]}")


learnedTrees = """Tree 0:
tail | 
gray | small | 
meatteeth | horns | hands | bulbous | 
hands | patches | patches | chewteeth | horns | lean | bulbous | longneck | 
paws | spots | brown | bulbous | bulbous | longleg | chewteeth | horns | bulbous | bulbous | toughskin | horns | hairless | stripes | patches | bulbous | 
walrus | humpbac+whale | spider+monkey | killer+whale | spider+monkey | dolphin | spider+monkey | killer+whale | deer | giraffe | rhinocerus | buffalo | sheep | dalmatian | horse | rabbit | otter | persian | weasel | giant+panda | squirre | hamster | bat | raccoon | polar+bear | grizzly+bear | wolf | lion | siamese+cat | german+shepherd | grizzly+bear | persian | 
---------
Tree 1:
small | 
chewteeth | paws | 
yellow | blue | paws | paws | 
longneck | black | paws | pads | stripes | paws | toughskin | toughskin | 
tail | stripes | horns | hands | paws | toughskin | red | orange | buckteeth | furry | strainteeth | chewteeth | hooves | pads | paws | pads | 
hippopotamus | skunk | chihuauah | fox | sheep | tiger | buffalo | lion | humpbac+whale | dolphin | blue+whale | polar+bear | seal | seal | otter | otter | cow | antelope | deer | giraffe | dalmatian | collie | leopard | siamese+cat | rabbit | hamster | pig | horse | giant+panda | german+shepherd | raccoon | siamese+cat | 
---------
Tree 2:
black | 
longneck | small | 
lean | toughskin | paws | longneck | 
hairless | longneck | flippers | buckteeth | brown | hands | hands | red | 
hooves | strainteeth | chewteeth | small | flippers | hands | pads | patches | longneck | lean | hooves | blue | paws | longneck | bulbous | strainteeth | 
hippopotamus | dolphin | elephant | walrus | antelope | deer | collie | spider+monkey | persian | moose | polar+bear | squirre | lion | lion | beaver | fox | seal | humpbac+whale | chihuauah | chihuauah | rat | skunk | skunk | leopard | sheep | otter | weasel | raccoon | zebra | ox | chimpan | grizzly+bear | 
---------
Tree 3:
gray | 
yellow | horns | 
longleg | stripes | hands | hands | 
chewteeth | longleg | longleg | strainteeth | flippers | longneck | red | chewteeth | 
hands | small | furry | lean | red | horns | strainteeth | tail | patches | horns | longleg | hands | small | orange | chewteeth | horns | 
squirre | otter | skunk | squirre | rhinocerus | humpbac+whale | elephant | deer | lion | fox | moose | polar+bear | grizzly+bear | gorilla | raccoon | wolf | sheep | cow | collie | dalmatian | horse | hamster | hamster | hamster | rabbit | giant+panda | mouse | bat | mouse | raccoon | chimpan | moose | 
---------
Tree 4:
brown | 
pads | paws | 
small | longneck | longleg | longleg | 
hands | paws | pads | paws | tail | paws | hooves | hands | 
chewteeth | hairless | black | paws | flippers | strainteeth | small | small | yellow | meatteeth | chewteeth | tail | buckteeth | paws | paws | meatteeth | 
rhinocerus | squirre | moose | elephant | ox | buffalo | rabbit | gorilla | persian | dolphin | giant+panda | grizzly+bear | humpbac+whale | seal | polar+bear | polar+bear | german+shepherd | fox | antelope | bat | killer+whale | wolf | killer+whale | otter | lion | giraffe | zebra | chimpan | zebra | zebra | chimpan | tiger | 
---------
Tree 5:
chewteeth | 
big | flippers | 
meatteeth | buckteeth | blue | longleg | 
meatteeth | small | furry | horns | orange | furry | big | furry | 
toughskin | longneck | hairless | meatteeth | pads | black | lean | stripes | longleg | tail | longneck | furry | white | longleg | longleg | meatteeth | 
squirre | rabbit | dolphin | spider+monkey | elephant | giant+panda | seal | humpbac+whale | collie | bobcat | german+shepherd | polar+bear | bat | gorilla | tiger | fox | pig | horse | pig | pig | cow | moose | deer | horse | tiger | lion | fox | rat | giraffe | giraffe | giraffe | zebra | 
---------
Tree 6:
red | 
orange | hairless | 
brown | gray | hairless | horns | 
hairless | hooves | tail | strainteeth | chewteeth | small | chewteeth | flippers | 
red | strainteeth | strainteeth | hands | hairless | buckteeth | patches | horns | longneck | longneck | longneck | big | chewteeth | horns | hands | black | 
sheep | elephant | persian | polar+bear | seal | horse | squirre | lion | raccoon | beaver | tiger | leopard | weasel | wolf | rat | leopard | humpbac+whale | rhinocerus | buffalo | giraffe | ox | cow | giraffe | moose | antelope | polar+bear | moose | deer | deer | cow | cow | horse | 
---------
Tree 7:
toughskin | 
buckteeth | orange | 
hooves | buckteeth | orange | orange | 
chewteeth | big | horns | pads | white | hairless | hairless | hairless | 
tail | paws | chewteeth | yellow | orange | gray | horns | tail | orange | orange | flippers | tail | hooves | tail | hairless | stripes | 
giraffe | deer | dolphin | humpbac+whale | killer+whale | buffalo | rhinocerus | pig | german+shepherd | rabbit | giant+panda | chimpan | horse | zebra | antelope | ox | beaver | bat | mouse | beaver | mouse | mouse | grizzly+bear | mouse | bobcat | squirre | grizzly+bear | rat | mole | squirre | hamster | raccoon | 
---------
Tree 8:
longleg | 
big | chewteeth | 
flippers | red | meatteeth | meatteeth | 
meatteeth | white | meatteeth | meatteeth | gray | meatteeth | furry | furry | 
spots | furry | hands | meatteeth | longleg | lean | hairless | pads | stripes | chewteeth | chewteeth | hairless | small | spots | meatteeth | chewteeth | 
zebra | bobcat | chihuauah | tiger | persian | rhinocerus | elephant | lion | pig | giraffe | killer+whale | rat | cow | collie | horse | leopard | squirre | spider+monkey | gorilla | grizzly+bear | polar+bear | gorilla | polar+bear | polar+bear | grizzly+bear | grizzly+bear | polar+bear | polar+bear | grizzly+bear | bat | bat | polar+bear | 
---------
Tree 9:
furry | 
furry | strainteeth | 
brown | toughskin | big | tail | 
strainteeth | meatteeth | meatteeth | bulbous | small | longneck | lean | tail | 
paws | stripes | patches | paws | horns | red | bulbous | bulbous | black | red | paws | hairless | chewteeth | paws | horns | hooves | 
humpbac+whale | sheep | dolphin | raccoon | squirre | ox | beaver | rat | collie | rabbit | mouse | ox | fox | seal | grizzly+bear | otter | antelope | moose | horse | dalmatian | otter | weasel | rabbit | german+shepherd | chimpan | leopard | elephant | german+shepherd | zebra | giraffe | giraffe | horse | 
---------
Tree 10:
hooves | 
strainteeth | hooves | 
longleg | pads | blue | hooves | 
longleg | flippers | hairless | small | flippers | patches | strainteeth | lean | 
small | strainteeth | black | patches | hooves | flippers | small | gray | flippers | flippers | big | flippers | longleg | hands | hooves | hooves | 
sheep | skunk | hippopotamus | grizzly+bear | squirre | gorilla | seal | moose | ox | rabbit | elephant | elephant | zebra | rabbit | rhinocerus | giant+panda | zebra | lion | horse | deer | collie | weasel | dalmatian | leopard | chihuauah | siamese+cat | tiger | bobcat | giraffe | polar+bear | giraffe | fox | 
---------
Tree 11:
flippers | 
white | longleg | 
big | flippers | yellow | patches | 
pads | hands | flippers | flippers | orange | small | strainteeth | black | 
stripes | gray | horns | pads | chewteeth | bulbous | buckteeth | blue | longneck | pads | stripes | longleg | horns | buckteeth | pads | pads | 
chihuauah | bat | bobcat | persian | otter | siamese+cat | fox | raccoon | sheep | raccoon | siamese+cat | mouse | sheep | sheep | hamster | persian | seal | dolphin | walrus | elephant | zebra | deer | horse | chimpan | collie | grizzly+bear | leopard | giant+panda | tiger | polar+bear | moose | german+shepherd | 
---------
Tree 12:
bulbous | 
black | bulbous | 
red | bulbous | toughskin | bulbous | 
bulbous | longleg | longleg | buckteeth | longleg | bulbous | hairless | hairless | 
tail | white | furry | hands | flippers | yellow | paws | meatteeth | chewteeth | chewteeth | stripes | longleg | furry | toughskin | bulbous | yellow | 
bobcat | fox | deer | horse | spider+monkey | bat | leopard | chimpan | buffalo | humpbac+whale | otter | grizzly+bear | gorilla | hippopotamus | elephant | lion | humpbac+whale | giant+panda | collie | wolf | sheep | skunk | ox | horse | zebra | raccoon | horse | tiger | horse | hamster | rat | rabbit | 
---------
Tree 13:
toughskin | 
longleg | paws | 
blue | hooves | red | paws | 
strainteeth | hooves | hooves | white | small | paws | tail | spots | 
stripes | lean | gray | furry | red | red | chewteeth | yellow | red | red | hairless | buckteeth | longneck | buckteeth | paws | paws | 
giraffe | cow | chihuauah | german+shepherd | squirre | hamster | pig | deer | raccoon | skunk | tiger | raccoon | zebra | tiger | skunk | raccoon | hippopotamus | dolphin | blue+whale | killer+whale | polar+bear | polar+bear | otter | tiger | walrus | beaver | seal | otter | walrus | zebra | deer | fox | 
---------
Tree 14:
buckteeth | 
meatteeth | toughskin | 
gray | furry | bulbous | chewteeth | 
bulbous | meatteeth | tail | chewteeth | hairless | toughskin | tail | longleg | 
paws | patches | bulbous | hairless | chewteeth | small | paws | red | longleg | hands | strainteeth | tail | hairless | meatteeth | toughskin | toughskin | 
humpbac+whale | cow | rhinocerus | giraffe | collie | leopard | beaver | polar+bear | persian | persian | sheep | rhinocerus | chihuauah | hamster | hamster | siamese+cat | fox | moose | bat | rat | otter | beaver | otter | otter | chimpan | rabbit | moose | chimpan | giant+panda | rabbit | tiger | lion | 
---------
Tree 15:
meatteeth | 
longleg | longneck | 
chewteeth | pads | paws | paws | 
lean | black | lean | bulbous | yellow | toughskin | flippers | lean | 
longleg | toughskin | furry | hairless | hands | tail | longneck | longleg | bulbous | bulbous | longleg | flippers | brown | furry | furry | tail | 
dolphin | elephant | sheep | zebra | persian | lion | bobcat | grizzly+bear | tiger | gorilla | otter | spider+monkey | beaver | squirre | rat | fox | sheep | raccoon | collie | raccoon | dalmatian | antelope | killer+whale | seal | collie | german+shepherd | cow | polar+bear | moose | gorilla | rabbit | giant+panda | 
---------
Tree 16:
chewteeth | 
gray | longneck | 
tail | strainteeth | hooves | paws | 
orange | bulbous | hairless | paws | tail | tail | tail | buckteeth | 
strainteeth | stripes | orange | furry | chewteeth | flippers | pads | blue | meatteeth | hairless | hairless | bulbous | pads | horns | longneck | hairless | 
otter | wolf | chihuauah | collie | zebra | horse | gorilla | deer | mole | squirre | beaver | fox | mouse | raccoon | rat | skunk | cow | moose | seal | humpbac+whale | giraffe | bat | giraffe | bat | persian | siamese+cat | rabbit | giant+panda | polar+bear | hamster | lion | rabbit | 
---------
Tree 17:
lean | 
white | big | 
orange | lean | strainteeth | paws | 
paws | lean | toughskin | longleg | buckteeth | big | paws | big | 
blue | paws | lean | toughskin | buckteeth | paws | horns | furry | buckteeth | longleg | bulbous | lean | stripes | spots | paws | hairless | 
squirre | weasel | mole | chihuauah | bat | weasel | otter | otter | sheep | persian | rabbit | hamster | siamese+cat | squirre | siamese+cat | siamese+cat | rhinocerus | zebra | dalmatian | polar+bear | horse | leopard | moose | cow | chimpan | spider+monkey | seal | chimpan | beaver | bobcat | chimpan | beaver | 
---------
Tree 18:
flippers | 
stripes | spots | 
toughskin | horns | meatteeth | hooves | 
hooves | red | hooves | red | paws | hands | stripes | big | 
tail | black | spots | hands | red | strainteeth | bulbous | bulbous | red | red | bulbous | buckteeth | longleg | furry | pads | pads | 
collie | mouse | bat | raccoon | german+shepherd | wolf | deer | horse | dolphin | seal | dolphin | seal | german+shepherd | bat | dolphin | polar+bear | ox | gorilla | hamster | giant+panda | humpbac+whale | otter | beaver | hippopotamus | polar+bear | polar+bear | lion | grizzly+bear | persian | moose | grizzly+bear | polar+bear | 
---------
Tree 19:
horns | 
longleg | pads | 
spots | toughskin | lean | lean | 
bulbous | paws | lean | longleg | hooves | horns | hairless | meatteeth | 
blue | hairless | longleg | lean | longneck | yellow | patches | patches | hands | hands | pads | horns | big | longneck | lean | yellow | 
humpbac+whale | hippopotamus | cow | giraffe | sheep | moose | grizzly+bear | chihuauah | skunk | zebra | pig | moose | giant+panda | deer | grizzly+bear | hamster | gorilla | zebra | antelope | chimpan | german+shepherd | wolf | killer+whale | polar+bear | otter | collie | persian | hamster | seal | bat | bobcat | bat | 
---------
Tree 20:
yellow | 
longneck | horns | 
lean | horns | meatteeth | horns | 
red | small | yellow | horns | flippers | horns | buckteeth | chewteeth | 
gray | hooves | black | bulbous | horns | strainteeth | big | gray | hairless | strainteeth | hands | strainteeth | small | patches | bulbous | hands | 
squirre | collie | raccoon | killer+whale | persian | pig | walrus | hamster | sheep | cow | horse | buffalo | rat | rabbit | grizzly+bear | rat | dalmatian | dolphin | antelope | horse | chihuauah | weasel | siamese+cat | german+shepherd | chimpan | tiger | fox | zebra | leopard | giraffe | lion | lion | 
---------
Tree 21:
black | 
lean | lean | 
brown | longneck | stripes | flippers | 
hooves | lean | lean | strainteeth | pads | longleg | chewteeth | lean | 
toughskin | hooves | big | longleg | flippers | lean | hooves | furry | lean | lean | white | strainteeth | meatteeth | brown | small | hands | 
rhinocerus | hippopotamus | hippopotamus | polar+bear | squirre | raccoon | walrus | beaver | squirre | giraffe | deer | lion | dolphin | persian | giraffe | fox | buffalo | humpbac+whale | grizzly+bear | chihuauah | gorilla | seal | wolf | grizzly+bear | horse | tiger | chimpan | hamster | seal | raccoon | otter | siamese+cat | 
---------
Tree 22:
big | 
big | lean | 
horns | patches | hooves | hooves | 
buckteeth | hooves | hooves | hooves | hands | small | orange | longleg | 
hooves | buckteeth | meatteeth | tail | hooves | bulbous | toughskin | hands | meatteeth | bulbous | stripes | chewteeth | hooves | lean | lean | hooves | 
buffalo | rhinocerus | hippopotamus | moose | sheep | giant+panda | persian | ox | walrus | persian | elephant | raccoon | gorilla | giant+panda | lion | pig | rabbit | collie | chihuauah | squirre | cow | seal | killer+whale | polar+bear | grizzly+bear | dalmatian | german+shepherd | lion | deer | giraffe | zebra | horse | 
---------
Tree 23:
big | 
big | pads | 
furry | buckteeth | orange | stripes | 
tail | longleg | red | chewteeth | strainteeth | white | flippers | hairless | 
hooves | paws | black | longleg | meatteeth | toughskin | small | big | meatteeth | hands | bulbous | stripes | big | brown | paws | stripes | 
sheep | persian | buffalo | moose | rabbit | squirre | giant+panda | horse | ox | moose | horse | bat | hippopotamus | humpbac+whale | dolphin | hippopotamus | rhinocerus | killer+whale | german+shepherd | leopard | fox | bobcat | german+shepherd | tiger | chihuauah | chimpan | antelope | horse | giraffe | deer | dalmatian | pig | 
---------
Tree 24:
small | 
meatteeth | patches | 
flippers | paws | flippers | flippers | 
spots | hands | hands | flippers | flippers | tail | bulbous | bulbous | 
flippers | longneck | big | lean | flippers | yellow | flippers | stripes | tail | buckteeth | bulbous | chewteeth | flippers | furry | pads | bulbous | 
elephant | giant+panda | sheep | ox | hippopotamus | walrus | humpbac+whale | humpbac+whale | buffalo | pig | giraffe | giant+panda | rhinocerus | moose | grizzly+bear | chihuauah | seal | antelope | squirre | rabbit | persian | dalmatian | bobcat | wolf | german+shepherd | tiger | mouse | horse | polar+bear | gorilla | lion | chimpan | 
---------
Tree 25:
buckteeth | 
meatteeth | meatteeth | 
hooves | blue | big | big | 
longleg | longneck | meatteeth | meatteeth | bulbous | horns | chewteeth | spots | 
paws | buckteeth | patches | flippers | hairless | yellow | big | white | hairless | meatteeth | orange | paws | tail | toughskin | small | bulbous | 
elephant | blue+whale | pig | cow | moose | seal | antelope | dolphin | chihuauah | rabbit | dalmatian | collie | beaver | hamster | leopard | raccoon | gorilla | chimpan | grizzly+bear | giant+panda | giant+panda | polar+bear | chimpan | chimpan | spider+monkey | chimpan | squirre | squirre | squirre | bat | squirre | bat | 
---------
Tree 26:
toughskin | 
chewteeth | yellow | 
chewteeth | toughskin | meatteeth | horns | 
bulbous | furry | lean | flippers | hands | flippers | flippers | furry | 
small | orange | black | tail | buckteeth | bulbous | yellow | paws | flippers | flippers | chewteeth | big | hands | small | longneck | bulbous | 
sheep | mouse | persian | hamster | rabbit | raccoon | hamster | bat | hippopotamus | polar+bear | ox | buffalo | elephant | seal | gorilla | grizzly+bear | chihuauah | mole | collie | horse | chihuauah | otter | mole | rat | antelope | giraffe | horse | spider+monkey | bobcat | lion | german+shepherd | fox | 
---------
Tree 27:
chewteeth | 
paws | bulbous | 
hands | tail | pads | pads | 
hooves | furry | furry | toughskin | hooves | white | paws | paws | 
hands | horns | hooves | stripes | chewteeth | strainteeth | meatteeth | small | big | hooves | bulbous | brown | gray | horns | paws | strainteeth | 
killer+whale | leopard | giant+panda | raccoon | siamese+cat | deer | squirre | grizzly+bear | humpbac+whale | seal | hippopotamus | deer | dalmatian | fox | killer+whale | killer+whale | elephant | sheep | giraffe | pig | walrus | horse | lion | chimpan | rhinocerus | polar+bear | tiger | skunk | chihuauah | ox | german+shepherd | ox | 
---------
Tree 28:
longneck | 
bulbous | tail | 
big | patches | chewteeth | toughskin | 
horns | patches | black | strainteeth | longneck | buckteeth | hooves | spots | 
white | furry | flippers | yellow | toughskin | small | gray | brown | toughskin | toughskin | lean | longneck | horns | buckteeth | furry | furry | 
elephant | horse | humpbac+whale | dolphin | grizzly+bear | tiger | hamster | german+shepherd | rabbit | hamster | collie | chimpan | siamese+cat | squirre | beaver | lion | dalmatian | seal | killer+whale | killer+whale | pig | persian | cow | giraffe | pig | giant+panda | moose | raccoon | antelope | raccoon | deer | bobcat | 
---------
Tree 29:
horns | 
meatteeth | big | 
meatteeth | strainteeth | gray | tail | 
chewteeth | stripes | meatteeth | big | toughskin | longleg | strainteeth | stripes | 
meatteeth | longneck | hands | hairless | paws | stripes | hands | orange | flippers | stripes | big | spots | furry | paws | furry | white | 
zebra | horse | collie | squirre | chihuauah | wolf | bobcat | gorilla | rhinocerus | lion | cow | deer | giraffe | bobcat | giraffe | giraffe | beaver | dolphin | killer+whale | killer+whale | leopard | fox | otter | otter | blue+whale | seal | persian | polar+bear | giant+panda | hippopotamus | hippopotamus | leopard | 
---------
Tree 30:
paws | 
hands | longleg | 
flippers | hairless | big | big | 
strainteeth | hands | hairless | chewteeth | flippers | hooves | buckteeth | hands | 
toughskin | paws | hands | yellow | buckteeth | bulbous | big | white | meatteeth | bulbous | brown | chewteeth | blue | flippers | hairless | chewteeth | 
sheep | hippopotamus | walrus | walrus | giraffe | cow | elephant | horse | squirre | collie | deer | dalmatian | rabbit | hamster | persian | skunk | buffalo | hamster | pig | raccoon | dolphin | polar+bear | humpbac+whale | beaver | polar+bear | gorilla | siamese+cat | chimpan | german+shepherd | leopard | german+shepherd | fox | 
---------
Tree 31:
gray | 
longleg | hooves | 
furry | pads | hooves | bulbous | 
chewteeth | small | big | lean | hands | flippers | spots | chewteeth | 
paws | black | horns | hooves | tail | strainteeth | chewteeth | pads | tail | toughskin | horns | furry | lean | buckteeth | bulbous | strainteeth | 
dolphin | collie | seal | wolf | squirre | tiger | siamese+cat | bat | giraffe | deer | antelope | giraffe | zebra | raccoon | horse | weasel | humpbac+whale | killer+whale | pig | grizzly+bear | walrus | gorilla | otter | lion | rabbit | cow | moose | hamster | elephant | rhinocerus | giant+panda | leopard | 
---------
Tree 32:
buckteeth | 
bulbous | longneck | 
buckteeth | tail | paws | paws | 
longneck | paws | paws | chewteeth | stripes | white | tail | strainteeth | 
hairless | bulbous | meatteeth | longneck | longneck | paws | paws | paws | paws | hooves | red | hooves | meatteeth | hooves | hooves | small | 
sheep | buffalo | antelope | cow | gorilla | horse | zebra | chimpan | elephant | dolphin | seal | walrus | elephant | bat | bat | bat | chihuauah | hamster | moose | moose | giant+panda | hippopotamus | persian | raccoon | polar+bear | otter | polar+bear | grizzly+bear | otter | collie | german+shepherd | wolf | 
---------
Tree 33:
big | 
chewteeth | strainteeth | 
horns | longleg | hands | hands | 
hooves | chewteeth | spots | chewteeth | longneck | strainteeth | buckteeth | bulbous | 
paws | brown | big | strainteeth | buckteeth | red | hands | paws | horns | big | paws | horns | hairless | stripes | chewteeth | flippers | 
weasel | tiger | squirre | antelope | elephant | lion | skunk | buffalo | hippopotamus | fox | deer | leopard | humpbac+whale | polar+bear | walrus | otter | sheep | chihuauah | pig | cow | ox | giant+panda | chihuauah | giant+panda | collie | rabbit | german+shepherd | chimpan | cow | pig | horse | horse | 
---------
Tree 34:
hands | 
gray | pads | 
stripes | hands | strainteeth | strainteeth | 
chewteeth | hands | strainteeth | red | hooves | meatteeth | horns | buckteeth | 
strainteeth | buckteeth | paws | pads | horns | white | big | strainteeth | paws | paws | big | big | tail | paws | pads | furry | 
dolphin | mole | tiger | german+shepherd | elephant | persian | seal | polar+bear | deer | horse | giraffe | buffalo | sheep | rhinocerus | cow | horse | lion | sheep | deer | moose | moose | wolf | chimpan | lion | bat | bat | bobcat | grizzly+bear | moose | raccoon | raccoon | raccoon | 
---------
Tree 35:
toughskin | 
hooves | longleg | 
lean | orange | hooves | orange | 
longneck | tail | tail | orange | bulbous | patches | strainteeth | chewteeth | 
gray | paws | furry | buckteeth | strainteeth | bulbous | longleg | horns | strainteeth | orange | chewteeth | spots | red | paws | longleg | orange | 
humpbac+whale | sheep | dolphin | horse | persian | german+shepherd | hippopotamus | rhinocerus | lion | tiger | deer | tiger | otter | mouse | fox | rabbit | giant+panda | persian | giant+panda | bobcat | moose | raccoon | cow | bobcat | hippopotamus | squirre | giant+panda | bat | raccoon | chimpan | gorilla | raccoon | 
---------
Tree 36:
lean | 
lean | small | 
strainteeth | spots | pads | meatteeth | 
stripes | hairless | strainteeth | pads | furry | meatteeth | hooves | hooves | 
strainteeth | horns | strainteeth | small | strainteeth | hooves | red | bulbous | brown | strainteeth | tail | buckteeth | gray | lean | longneck | hooves | 
buffalo | blue+whale | sheep | rabbit | deer | horse | chimpan | gorilla | moose | giant+panda | squirre | rhinocerus | collie | tiger | leopard | bobcat | dolphin | killer+whale | otter | beaver | gorilla | otter | horse | rabbit | cow | moose | seal | walrus | polar+bear | polar+bear | polar+bear | polar+bear | 
---------
Tree 37:
hooves | 
paws | chewteeth | 
longleg | stripes | yellow | pads | 
bulbous | orange | horns | pads | longleg | lean | lean | black | 
flippers | buckteeth | longneck | pads | meatteeth | strainteeth | hairless | hairless | strainteeth | strainteeth | flippers | flippers | hairless | brown | lean | longleg | 
weasel | chihuauah | fox | bobcat | mouse | beaver | squirre | squirre | siamese+cat | hamster | persian | hamster | sheep | sheep | bat | rabbit | elephant | ox | seal | tiger | giraffe | cow | german+shepherd | wolf | zebra | leopard | leopard | polar+bear | wolf | deer | gorilla | giant+panda | 
---------
Tree 38:
toughskin | 
brown | yellow | 
toughskin | meatteeth | pads | chewteeth | 
tail | spots | hooves | bulbous | toughskin | hooves | horns | hands | 
buckteeth | hooves | horns | longneck | bulbous | bulbous | longleg | flippers | hands | hands | hands | chewteeth | gray | longleg | meatteeth | hooves | 
rhinocerus | grizzly+bear | blue+whale | dolphin | collie | persian | seal | polar+bear | sheep | moose | horse | cow | german+shepherd | bobcat | german+shepherd | wolf | chimpan | giant+panda | rat | otter | tiger | lion | leopard | fox | walrus | hamster | hamster | hamster | bat | mouse | rabbit | bat | 
---------
Tree 39:
lean | 
white | lean | 
pads | toughskin | paws | lean | 
brown | longneck | patches | spots | bulbous | lean | toughskin | toughskin | 
chewteeth | strainteeth | pads | longleg | lean | meatteeth | gray | orange | furry | hooves | pads | strainteeth | orange | big | lean | small | 
weasel | buffalo | hippopotamus | antelope | lion | leopard | leopard | giraffe | killer+whale | rat | skunk | zebra | polar+bear | wolf | raccoon | tiger | pig | dolphin | chimpan | horse | chihuauah | german+shepherd | dalmatian | siamese+cat | giant+panda | sheep | ox | giant+panda | cow | cow | hamster | horse | 
---------
Tree 40:
blue | 
pads | bulbous | 
big | patches | furry | small | 
tail | buckteeth | tail | small | longleg | hands | meatteeth | hooves | 
yellow | chewteeth | longleg | bulbous | small | hairless | pads | horns | longleg | small | buckteeth | longleg | horns | lean | longneck | hairless | 
dolphin | hippopotamus | spider+monkey | otter | killer+whale | otter | humpbac+whale | seal | squirre | dalmatian | weasel | gorilla | elephant | giant+panda | pig | grizzly+bear | rhinocerus | antelope | giraffe | giraffe | fox | bobcat | deer | moose | sheep | cow | rat | rabbit | zebra | tiger | horse | wolf | 
---------
Tree 41:
patches | 
strainteeth | longleg | 
longneck | buckteeth | longleg | chewteeth | 
longleg | strainteeth | strainteeth | strainteeth | buckteeth | strainteeth | horns | small | 
hairless | furry | meatteeth | strainteeth | longleg | white | orange | furry | blue | pads | meatteeth | meatteeth | meatteeth | strainteeth | longleg | strainteeth | 
persian | collie | squirre | skunk | elephant | mouse | rabbit | giant+panda | buffalo | moose | antelope | horse | antelope | zebra | rhinocerus | giraffe | walrus | lion | lion | grizzly+bear | polar+bear | tiger | bobcat | fox | german+shepherd | german+shepherd | siamese+cat | siamese+cat | otter | siamese+cat | leopard | grizzly+bear | 
---------
Tree 42:
hands | 
blue | paws | 
strainteeth | strainteeth | tail | paws | 
tail | big | flippers | paws | small | patches | patches | horns | 
buckteeth | chewteeth | flippers | lean | paws | tail | buckteeth | hairless | paws | tail | toughskin | flippers | hooves | red | patches | tail | 
dolphin | siamese+cat | bobcat | wolf | wolf | otter | humpbac+whale | killer+whale | squirre | rabbit | antelope | raccoon | moose | giant+panda | hippopotamus | hippopotamus | buffalo | rhinocerus | gorilla | bat | chimpan | giraffe | german+shepherd | lion | sheep | polar+bear | cow | skunk | tiger | mouse | hamster | horse | 
---------
Tree 43:
tail | 
tail | tail | 
patches | bulbous | flippers | paws | 
pads | longneck | longneck | flippers | horns | lean | longleg | red | 
toughskin | pads | tail | small | flippers | tail | brown | yellow | flippers | flippers | blue | patches | orange | furry | brown | flippers | 
bat | dolphin | otter | rabbit | cow | killer+whale | beaver | rabbit | cow | collie | hamster | squirre | deer | lion | wolf | fox | sheep | buffalo | mole | giant+panda | walrus | walrus | grizzly+bear | polar+bear | dolphin | giraffe | seal | persian | ox | seal | humpbac+whale | moose | 
---------
Tree 44:
red | 
hairless | flippers | 
chewteeth | stripes | flippers | lean | 
buckteeth | meatteeth | meatteeth | flippers | gray | longleg | longneck | meatteeth | 
blue | hooves | meatteeth | strainteeth | longneck | stripes | pads | horns | spots | brown | spots | patches | toughskin | pads | flippers | stripes | 
pig | grizzly+bear | moose | cow | humpbac+whale | chihuauah | moose | humpbac+whale | giraffe | rhinocerus | giraffe | elephant | cow | giant+panda | humpbac+whale | ox | zebra | dolphin | tiger | gorilla | antelope | leopard | dalmatian | horse | squirre | mouse | collie | bobcat | rabbit | otter | chimpan | siamese+cat | 
---------
Tree 45:
meatteeth | 
buckteeth | longleg | 
furry | lean | small | yellow | 
bulbous | patches | patches | tail | furry | strainteeth | black | small | 
lean | stripes | big | yellow | horns | orange | hooves | lean | longneck | big | flippers | small | flippers | yellow | yellow | blue | 
raccoon | giraffe | otter | lion | sheep | chihuauah | giant+panda | collie | german+shepherd | siamese+cat | deer | wolf | squirre | fox | rat | hamster | hippopotamus | dolphin | grizzly+bear | gorilla | pig | zebra | moose | grizzly+bear | moose | buffalo | bat | polar+bear | cow | ox | horse | chimpan | 
---------
Tree 46:
furry | 
spots | spots | 
chewteeth | flippers | small | small | 
paws | blue | toughskin | big | small | spots | red | pads | 
longleg | paws | gray | spots | bulbous | spots | hooves | spots | lean | patches | spots | lean | horns | flippers | spots | spots | 
killer+whale | killer+whale | hippopotamus | hippopotamus | rhinocerus | elephant | zebra | horse | gorilla | grizzly+bear | chimpan | squirre | weasel | tiger | wolf | lion | antelope | squirre | buffalo | wolf | humpbac+whale | seal | polar+bear | fox | collie | rabbit | leopard | deer | pig | horse | ox | raccoon | 
---------
Tree 47:
chewteeth | 
hands | lean | 
hooves | chewteeth | strainteeth | strainteeth | 
buckteeth | buckteeth | bulbous | bulbous | stripes | buckteeth | small | bulbous | 
furry | gray | white | small | buckteeth | bulbous | hands | stripes | orange | pads | buckteeth | hands | hands | chewteeth | yellow | bulbous | 
rabbit | dolphin | beaver | beaver | seal | blue+whale | sheep | persian | otter | siamese+cat | dolphin | bat | killer+whale | moose | otter | otter | cow | buffalo | collie | hamster | gorilla | elephant | deer | horse | lion | grizzly+bear | polar+bear | lion | wolf | fox | tiger | german+shepherd | 
---------
Tree 48:
hands | 
tail | meatteeth | 
furry | brown | bulbous | orange | 
pads | tail | paws | orange | longneck | longleg | chewteeth | yellow | 
small | chewteeth | hairless | buckteeth | orange | orange | horns | red | hairless | orange | tail | horns | spots | brown | toughskin | orange | 
sheep | weasel | collie | giraffe | sheep | chihuauah | mouse | squirre | elephant | pig | dalmatian | gorilla | deer | giraffe | wolf | bobcat | grizzly+bear | killer+whale | walrus | grizzly+bear | walrus | hamster | walrus | grizzly+bear | seal | dolphin | hippopotamus | humpbac+whale | persian | polar+bear | leopard | giant+panda | 
---------
Tree 49:
toughskin | 
stripes | longneck | 
strainteeth | hooves | buckteeth | pads | 
meatteeth | hands | chewteeth | paws | meatteeth | pads | toughskin | meatteeth | 
big | horns | tail | tail | longneck | longleg | buckteeth | furry | white | tail | buckteeth | toughskin | strainteeth | horns | tail | bulbous | 
ox | sheep | chihuauah | buffalo | dolphin | seal | seal | polar+bear | buffalo | german+shepherd | dalmatian | siamese+cat | moose | antelope | horse | zebra | gorilla | deer | squirre | spider+monkey | hamster | mole | fox | raccoon | lion | lion | gorilla | lion | giant+panda | leopard | chimpan | rabbit | 
---------"""

learnedTrees = learnedTrees.split("---------\n")
setje = set()

#for treeStr in learnedTrees:
#    lines = treeStr.splitlines()
#    print(lines[0])
#    root = Node(lines[1].split(" | ")[0])
#    root.left = Node(lines[2].split(" | ")[0])
#    root.right = Node(lines[2].split(" | ")[1])
#    root.left.left = Node(lines[3].split(" | ")[0])
#    root.left.right = Node(lines[3].split(" | ")[1])
#    root.right.left = Node(lines[3].split(" | ")[2])
#    root.right.right = Node(lines[3].split(" | ")[3])
#    root.left.left.left = Node(lines[4].split(" | ")[0])
#    root.left.left.right = Node(lines[4].split(" | ")[1])
#    root.left.right.left = Node(lines[4].split(" | ")[2])
#    root.left.right.right = Node(lines[4].split(" | ")[3])
#    root.right.left.left = Node(lines[4].split(" | ")[4])
#    root.right.left.right = Node(lines[4].split(" | ")[5])
#    root.right.right.left = Node(lines[4].split(" | ")[6])
#    root.right.right.right = Node(lines[4].split(" | ")[7])
#    root.left.left.left.left = Node(lines[5].split(" | ")[0])
#    root.left.left.left.right = Node(lines[5].split(" | ")[1])
#    root.left.left.right.left = Node(lines[5].split(" | ")[2])
#    root.left.left.right.right = Node(lines[5].split(" | ")[3])
#    root.left.right.left.left = Node(lines[5].split(" | ")[4])
#    root.left.right.left.right = Node(lines[5].split(" | ")[5])
#    root.left.right.right.left = Node(lines[5].split(" | ")[6])
#    root.left.right.right.right = Node(lines[5].split(" | ")[7])
#    root.right.left.left.left = Node(lines[5].split(" | ")[8])
#    root.right.left.left.right = Node(lines[5].split(" | ")[9])
#    root.right.left.right.left = Node(lines[5].split(" | ")[10])
#    root.right.left.right.right = Node(lines[5].split(" | ")[11])
#    root.right.right.left.left = Node(lines[5].split(" | ")[12])
#    root.right.right.left.right = Node(lines[5].split(" | ")[13])
#    root.right.right.right.left = Node(lines[5].split(" | ")[14])
#    root.right.right.right.right = Node(lines[5].split(" | ")[15])
#    root.left.left.left.left.left = Node(lines[6].split(" | ")[0])
#    root.left.left.left.left.right = Node(lines[6].split(" | ")[1])
#    root.left.left.left.right.left = Node(lines[6].split(" | ")[2])
#    root.left.left.left.right.right = Node(lines[6].split(" | ")[3])
#    root.left.left.right.left.left = Node(lines[6].split(" | ")[4])
#    root.left.left.right.left.right = Node(lines[6].split(" | ")[5])
#    root.left.left.right.right.left = Node(lines[6].split(" | ")[6])
#    root.left.left.right.right.right = Node(lines[6].split(" | ")[7])
#    root.left.right.left.left.left = Node(lines[6].split(" | ")[8])
#    root.left.right.left.left.right = Node(lines[6].split(" | ")[9])
#    root.left.right.left.right.left = Node(lines[6].split(" | ")[10])
#    root.left.right.left.right.right = Node(lines[6].split(" | ")[11])
#    root.left.right.right.left.left = Node(lines[6].split(" | ")[12])
#    root.left.right.right.left.right = Node(lines[6].split(" | ")[13])
#    root.left.right.right.right.left = Node(lines[6].split(" | ")[14])
#    root.left.right.right.right.right = Node(lines[6].split(" | ")[15])
#    root.right.left.left.left.left = Node(lines[6].split(" | ")[16])
#    root.right.left.left.left.right = Node(lines[6].split(" | ")[17])
#    root.right.left.left.right.left = Node(lines[6].split(" | ")[18])
#    root.right.left.left.right.right = Node(lines[6].split(" | ")[19])
#    root.right.left.right.left.left = Node(lines[6].split(" | ")[20])
#    root.right.left.right.left.right = Node(lines[6].split(" | ")[21])
#    root.right.left.right.right.left = Node(lines[6].split(" | ")[22])
#    root.right.left.right.right.right = Node(lines[6].split(" | ")[23])
#    root.right.right.left.left.left = Node(lines[6].split(" | ")[24])
#    root.right.right.left.left.right = Node(lines[6].split(" | ")[25])
#    root.right.right.left.right.left = Node(lines[6].split(" | ")[26])
#    root.right.right.left.right.right = Node(lines[6].split(" | ")[27])
#    root.right.right.right.left.left = Node(lines[6].split(" | ")[28])
#    root.right.right.right.left.right = Node(lines[6].split(" | ")[29])
#    root.right.right.right.right.left = Node(lines[6].split(" | ")[30])
#    root.right.right.right.right.right = Node(lines[6].split(" | ")[31])
#    print(root)
#


baseTrees = """Tree 0:
L7 | 
L4 | L4 | 
R6 | L3 | L7 | R0 | 
10 | 5 | 17 | 8 | 2 | 11 | 13 | 15 | 
---------
Tree 1:
L3 | 
R3 | R8 | 
L1 | R3 | L7 | R8 | 
12 | 14 | 13 | 12 | 12 | 13 | 12 | 13 | 
---------
Tree 2:
L4 | 
R0 | L1 | 
R0 | R9 | L1 | L1 | 
4 | 1 | 1 | 4 | 4 | 1 | 3 | 1 | 
---------
Tree 3:
R4 | 
L4 | L8 | 
L9 | L8 | R5 | R9 | 
17 | 13 | 7 | 7 | 8 | 5 | 15 | 12 | 
---------
Tree 4:
R5 | 
R5 | R9 | 
L3 | L8 | R6 | L0 | 
1 | 14 | 10 | 6 | 5 | 16 | 4 | 3 | 
---------
Tree 5:
R6 | 
R9 | R5 | 
L9 | R3 | L4 | R6 | 
1 | 1 | 3 | 1 | 1 | 7 | 15 | 1 | 
---------
Tree 6:
R4 | 
L9 | L6 | 
R4 | R3 | L3 | L9 | 
9 | 15 | 7 | 9 | 9 | 8 | 11 | 9 | 
---------
Tree 7:
L6 | 
L0 | L4 | 
R0 | L6 | R4 | L1 | 
16 | 5 | 7 | 1 | 9 | 5 | 17 | 9 | 
---------
Tree 8:
L8 | 
L6 | R8 | 
L4 | L5 | R3 | L4 | 
15 | 10 | 8 | 5 | 10 | 4 | 0 | 2 | 
---------
Tree 9:
L4 | 
L3 | R3 | 
L2 | L9 | R0 | L9 | 
15 | 11 | 16 | 18 | 7 | 8 | 10 | 17 | 
---------
Tree 10:
L5 | 
L2 | L0 | 
L0 | L3 | L1 | R1 | 
4 | 0 | 3 | 4 | 2 | 2 | 4 | 4 | 
---------
Tree 11:
L9 | 
R8 | R1 | 
R9 | L7 | R2 | R7 | 
5 | 8 | 13 | 10 | 0 | 4 | 9 | 4 | 
---------
Tree 12:
L7 | 
L7 | L5 | 
L8 | L1 | R8 | L6 | 
5 | 10 | 5 | 5 | 5 | 5 | 5 | 5 | 
---------
Tree 13:
L7 | 
R5 | L0 | 
R4 | L0 | L6 | L6 | 
1 | 13 | 12 | 9 | 18 | 4 | 6 | 0 | 
---------
Tree 14:
L2 | 
R7 | R4 | 
L7 | L2 | R5 | L8 | 
6 | 7 | 7 | 6 | 7 | 7 | 6 | 6 | 
---------
Tree 15:
L3 | 
L0 | R0 | 
L9 | R5 | L8 | L4 | 
7 | 5 | 5 | 4 | 7 | 8 | 5 | 2 | 
---------
Tree 16:
R8 | 
R6 | L2 | 
L9 | L1 | R1 | R5 | 
1 | 1 | 0 | 6 | 0 | 1 | 1 | 4 | 
---------
Tree 17:
L2 | 
L2 | R8 | 
L2 | R6 | L2 | R2 | 
8 | 8 | 8 | 8 | 8 | 8 | 8 | 2 | 
---------
Tree 18:
L9 | 
L3 | L6 | 
R9 | L9 | L7 | L4 | 
3 | 1 | 2 | 18 | 7 | 5 | 6 | 17 | 
---------
Tree 19:
R3 | 
L3 | L9 | 
R6 | L8 | R5 | L4 | 
6 | 13 | 5 | 9 | 15 | 8 | 13 | 15 | 
---------
Tree 20:
L6 | 
R3 | L0 | 
L7 | R8 | R3 | L7 | 
2 | 8 | 13 | 5 | 6 | 18 | 14 | 2 | 
---------
Tree 21:
R4 | 
R3 | L2 | 
R6 | R4 | L5 | R2 | 
0 | 13 | 10 | 11 | 10 | 15 | 9 | 12 | 
---------
Tree 22:
L0 | 
L4 | R9 | 
R0 | R0 | L7 | R0 | 
1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 
---------
Tree 23:
R4 | 
L4 | R9 | 
L8 | L8 | L7 | R8 | 
2 | 8 | 5 | 12 | 2 | 2 | 3 | 10 | 
---------
Tree 24:
L0 | 
R0 | R7 | 
R1 | R1 | L1 | R9 | 
11 | 11 | 11 | 11 | 11 | 10 | 11 | 11 | 
---------
Tree 25:
R7 | 
R3 | R5 | 
L4 | R7 | R4 | R8 | 
4 | 11 | 11 | 11 | 1 | 4 | 16 | 9 | 
---------
Tree 26:
L0 | 
L4 | R5 | 
L9 | R7 | L9 | R1 | 
6 | 8 | 4 | 7 | 7 | 16 | 12 | 7 | 
---------
Tree 27:
R1 | 
R3 | L0 | 
L4 | L3 | L4 | R2 | 
5 | 17 | 10 | 15 | 12 | 9 | 3 | 9 | 
---------
Tree 28:
R5 | 
L4 | R0 | 
L1 | R7 | R4 | L2 | 
18 | 5 | 6 | 6 | 5 | 8 | 7 | 14 | 
---------
Tree 29:
L9 | 
R2 | L4 | 
R7 | L8 | R5 | R2 | 
14 | 11 | 12 | 4 | 18 | 17 | 8 | 8 | 
---------
Tree 30:
L3 | 
L0 | R3 | 
R3 | L3 | R9 | L2 | 
10 | 14 | 11 | 6 | 13 | 9 | 13 | 9 | 
---------
Tree 31:
R0 | 
L8 | L5 | 
L0 | L0 | L6 | L2 | 
15 | 11 | 14 | 11 | 15 | 3 | 3 | 4 | 
---------
Tree 32:
L2 | 
L5 | R5 | 
R3 | R7 | R0 | R6 | 
14 | 11 | 8 | 5 | 15 | 13 | 8 | 12 | 
---------
Tree 33:
L0 | 
L1 | L8 | 
R1 | L9 | R3 | R1 | 
11 | 10 | 11 | 10 | 13 | 2 | 9 | 1 | 
---------
Tree 34:
R7 | 
R9 | R2 | 
R1 | L7 | L3 | L0 | 
15 | 17 | 16 | 16 | 16 | 16 | 16 | 16 | 
---------
Tree 35:
L0 | 
L2 | L1 | 
L1 | R5 | L9 | R0 | 
14 | 1 | 11 | 1 | 10 | 16 | 2 | 14 | 
---------
Tree 36:
R0 | 
R1 | L1 | 
L4 | R5 | L1 | R4 | 
3 | 4 | 2 | 2 | 3 | 2 | 1 | 11 | 
---------
Tree 37:
R9 | 
L2 | R9 | 
L4 | R5 | L3 | L8 | 
3 | 10 | 8 | 3 | 13 | 5 | 8 | 6 | 
---------
Tree 38:
L4 | 
R4 | R4 | 
R2 | L5 | R6 | R4 | 
16 | 12 | 17 | 5 | 7 | 14 | 16 | 0 | 
---------
Tree 39:
L2 | 
R2 | R0 | 
L0 | R3 | L2 | R9 | 
6 | 9 | 9 | 11 | 18 | 3 | 16 | 9 | 
---------
Tree 40:
L3 | 
R3 | L0 | 
L6 | L6 | R7 | R2 | 
9 | 10 | 10 | 9 | 9 | 10 | 9 | 9 | 
---------
Tree 41:
L9 | 
L5 | R5 | 
R7 | L2 | R5 | L6 | 
10 | 7 | 2 | 17 | 8 | 14 | 11 | 8 | 
---------
Tree 42:
L2 | 
R8 | R3 | 
L7 | R7 | R6 | L1 | 
11 | 10 | 0 | 17 | 4 | 12 | 10 | 1 | 
---------
Tree 43:
L0 | 
R6 | L3 | 
L1 | L4 | R7 | R1 | 
16 | 8 | 5 | 16 | 4 | 7 | 11 | 7 | 
---------
Tree 44:
R7 | 
L6 | L7 | 
L0 | L8 | L1 | L9 | 
9 | 4 | 18 | 14 | 2 | 6 | 7 | 16 | 
---------
Tree 45:
L8 | 
L6 | R8 | 
L1 | R4 | L5 | L4 | 
5 | 3 | 15 | 10 | 8 | 0 | 8 | 0 | 
---------
Tree 46:
L9 | 
R8 | L2 | 
L5 | L0 | R2 | R3 | 
10 | 1 | 16 | 8 | 8 | 9 | 3 | 6 | 
---------
Tree 47:
L4 | 
L4 | L8 | 
R5 | L0 | L2 | R9 | 
3 | 8 | 14 | 7 | 13 | 7 | 8 | 6 | 
---------
Tree 48:
R1 | 
R9 | R8 | 
R6 | R1 | L5 | L5 | 
17 | 9 | 0 | 13 | 16 | 15 | 13 | 9 | 
---------
Tree 49:
L6 | 
R4 | R8 | 
R4 | R3 | R3 | L7 | 
2 | 16 | 7 | 13 | 5 | 5 | 12 | 5 | 
"""

dropoutTrees = """Tree 0:
R0 | 
R4 | R5 | 
R6 | L3 | R3 | R0 | 
10 | 5 | 6 | 7 | 7 | 6 | 8 | 5 | 
---------
Tree 1:
R2 | 
L1 | R8 | 
L1 | R3 | L7 | R9 | 
12 | 14 | 13 | 12 | 12 | 13 | 13 | 12 | 
---------
Tree 2:
R9 | 
L9 | L6 | 
R0 | R9 | L9 | L6 | 
9 | 7 | 10 | 16 | 9 | 11 | 8 | 8 | 
---------
Tree 3:
R4 | 
L5 | L8 | 
L9 | L8 | R5 | R9 | 
17 | 13 | 7 | 7 | 8 | 5 | 5 | 12 | 
---------
Tree 4:
R5 | 
R4 | R9 | 
L3 | R8 | R6 | L0 | 
1 | 14 | 10 | 6 | 5 | 16 | 10 | 3 | 
---------
Tree 5:
L2 | 
L5 | R9 | 
L9 | R9 | L6 | R6 | 
4 | 16 | 6 | 13 | 6 | 7 | 11 | 14 | 
---------
Tree 6:
R9 | 
L6 | L6 | 
L2 | R8 | L2 | L9 | 
9 | 6 | 7 | 9 | 16 | 11 | 13 | 17 | 
---------
Tree 7:
R1 | 
L0 | L5 | 
R3 | R8 | L4 | L0 | 
12 | 5 | 7 | 6 | 9 | 5 | 6 | 9 | 
---------
Tree 8:
L5 | 
R1 | R8 | 
L4 | L4 | R3 | L4 | 
15 | 10 | 8 | 5 | 10 | 8 | 6 | 13 | 
---------
Tree 9:
R6 | 
L3 | R6 | 
L2 | L9 | R2 | L9 | 
10 | 11 | 8 | 7 | 7 | 8 | 6 | 7 | 
---------
Tree 10:
L5 | 
R6 | L0 | 
R9 | L9 | L1 | R1 | 
4 | 4 | 4 | 5 | 10 | 4 | 4 | 4 | 
---------
Tree 11:
L9 | 
R8 | R1 | 
R9 | L2 | R2 | R7 | 
8 | 15 | 13 | 10 | 16 | 10 | 9 | 16 | 
---------
Tree 12:
L7 | 
R9 | L5 | 
R6 | L1 | R8 | L7 | 
5 | 10 | 5 | 5 | 7 | 5 | 5 | 7 | 
---------
Tree 13:
L7 | 
R5 | L4 | 
R4 | L0 | R0 | R9 | 
9 | 5 | 9 | 7 | 7 | 7 | 6 | 8 | 
---------
Tree 14:
L9 | 
R7 | R6 | 
R8 | L2 | L0 | L8 | 
6 | 7 | 7 | 6 | 6 | 7 | 6 | 7 | 
---------
Tree 15:
R9 | 
L1 | R2 | 
L9 | R6 | R6 | R0 | 
7 | 17 | 9 | 7 | 9 | 6 | 5 | 9 | 
---------
Tree 16:
R8 | 
R6 | L2 | 
L9 | L1 | R1 | L6 | 
1 | 1 | 7 | 1 | 0 | 1 | 1 | 1 | 
---------
Tree 17:
L9 | 
R9 | L9 | 
R9 | L4 | R5 | R2 | 
8 | 8 | 8 | 8 | 8 | 8 | 9 | 2 | 
---------
Tree 18:
L9 | 
L3 | L9 | 
R9 | L9 | R1 | L4 | 
9 | 18 | 6 | 9 | 16 | 8 | 15 | 17 | 
---------
Tree 19:
L9 | 
R9 | L4 | 
R7 | L8 | L3 | R6 | 
4 | 13 | 18 | 9 | 15 | 12 | 13 | 15 | 
---------
Tree 20:
L6 | 
R1 | R8 | 
L7 | R1 | R3 | L7 | 
2 | 14 | 13 | 5 | 10 | 18 | 14 | 2 | 
---------
Tree 21:
R4 | 
R3 | L7 | 
R6 | R4 | L5 | R2 | 
16 | 13 | 10 | 11 | 6 | 15 | 9 | 12 | 
---------
Tree 22:
L7 | 
L2 | R9 | 
R9 | L9 | L7 | L7 | 
3 | 12 | 10 | 6 | 5 | 10 | 11 | 10 | 
---------
Tree 23:
L6 | 
L4 | R9 | 
L8 | L8 | L6 | L1 | 
12 | 8 | 5 | 8 | 7 | 7 | 15 | 10 | 
---------
Tree 24:
L0 | 
R0 | R7 | 
L0 | R1 | R2 | R9 | 
11 | 11 | 11 | 12 | 12 | 10 | 11 | 11 | 
---------
Tree 25:
R7 | 
R3 | R5 | 
L4 | R7 | R4 | R8 | 
18 | 6 | 11 | 11 | 1 | 4 | 16 | 9 | 
---------
Tree 26:
L0 | 
R0 | R8 | 
L9 | R7 | L9 | R2 | 
15 | 18 | 6 | 7 | 7 | 16 | 8 | 8 | 
---------
Tree 27:
R9 | 
R3 | L0 | 
L7 | L4 | R1 | L1 | 
12 | 8 | 10 | 7 | 17 | 9 | 8 | 9 | 
---------
Tree 28:
R5 | 
L4 | R0 | 
L1 | R7 | R4 | L2 | 
18 | 5 | 6 | 6 | 5 | 8 | 7 | 14 | 
---------
Tree 29:
R9 | 
R2 | L4 | 
L2 | L8 | R4 | R2 | 
9 | 7 | 6 | 10 | 16 | 17 | 12 | 12 | 
---------
Tree 30:
L3 | 
L0 | R3 | 
R0 | R3 | R9 | L9 | 
10 | 6 | 6 | 6 | 5 | 9 | 6 | 9 | 
---------
Tree 31:
R0 | 
R9 | L9 | 
L0 | L0 | L2 | L2 | 
9 | 7 | 17 | 10 | 6 | 6 | 10 | 4 | 
---------
Tree 32:
L2 | 
R4 | R5 | 
R3 | R7 | R0 | R6 | 
14 | 11 | 8 | 5 | 15 | 13 | 8 | 12 | 
---------
Tree 33:
L0 | 
L9 | R3 | 
R1 | L9 | L0 | R0 | 
11 | 9 | 10 | 10 | 7 | 7 | 9 | 10 | 
---------
Tree 34:
R7 | 
R9 | R8 | 
R1 | L7 | L9 | R3 | 
15 | 17 | 18 | 16 | 16 | 16 | 16 | 16 | 
---------
Tree 35:
L0 | 
L2 | L1 | 
L1 | R0 | L9 | R0 | 
14 | 1 | 7 | 6 | 1 | 16 | 2 | 8 | 
---------
Tree 36:
R0 | 
L3 | L2 | 
L4 | L9 | L1 | R9 | 
3 | 0 | 2 | 2 | 0 | 4 | 1 | 11 | 
---------
Tree 37:
R9 | 
L2 | L7 | 
L4 | R5 | L3 | L3 | 
8 | 5 | 5 | 6 | 17 | 11 | 15 | 12 | 
---------
Tree 38:
L3 | 
R4 | R7 | 
L4 | L5 | L8 | R4 | 
9 | 5 | 6 | 5 | 7 | 5 | 8 | 7 | 
---------
Tree 39:
L2 | 
R2 | R0 | 
L0 | R3 | L2 | R9 | 
9 | 9 | 9 | 11 | 6 | 3 | 16 | 9 | 
---------
Tree 40:
R0 | 
L6 | L0 | 
L6 | R6 | R4 | R2 | 
10 | 10 | 9 | 9 | 9 | 10 | 9 | 10 | 
---------
Tree 41:
L9 | 
L5 | R5 | 
R7 | R6 | R5 | L6 | 
10 | 7 | 1 | 7 | 8 | 14 | 10 | 8 | 
---------
Tree 42:
L0 | 
R0 | R3 | 
L7 | R7 | R6 | R0 | 
11 | 8 | 7 | 7 | 7 | 6 | 7 | 1 | 
---------
Tree 43:
L0 | 
R6 | L3 | 
L1 | R7 | R7 | L1 | 
16 | 8 | 5 | 16 | 4 | 7 | 15 | 7 | 
---------
Tree 44:
R0 | 
L6 | L7 | 
L0 | L8 | L3 | L9 | 
9 | 1 | 7 | 14 | 5 | 6 | 7 | 16 | 
---------
Tree 45:
L8 | 
L6 | R8 | 
L1 | R4 | L5 | L4 | 
18 | 15 | 15 | 10 | 8 | 0 | 8 | 0 | 
---------
Tree 46:
R4 | 
R1 | L1 | 
L5 | R2 | R0 | R3 | 
9 | 5 | 6 | 8 | 8 | 9 | 5 | 6 | 
---------
Tree 47:
L4 | 
L4 | L9 | 
R6 | L0 | L2 | L0 | 
3 | 8 | 5 | 7 | 6 | 7 | 15 | 6 | 
---------
Tree 48:
R1 | 
L3 | R8 | 
R6 | R1 | L5 | L5 | 
4 | 9 | 6 | 13 | 6 | 15 | 13 | 9 | 
---------
Tree 49:
R5 | 
R4 | R8 | 
R7 | R8 | R8 | R0 | 
12 | 9 | 7 | 13 | 5 | 5 | 12 | 10 | 
"""

twoStageTrees = """Tree 0:
R0 | 
R4 | L4 | 
R1 | L4 | L0 | L5 | 
9 | 6 | 7 | 8 | 5 | 0 | 4 | 5 | 
---------
Tree 1:
R6 | 
L6 | L1 | 
R7 | R1 | L2 | R8 | 
17 | 16 | 9 | 7 | 6 | 7 | 7 | 12 | 
---------
Tree 2:
R6 | 
L3 | R7 | 
R0 | L0 | R6 | L1 | 
16 | 2 | 10 | 8 | 2 | 11 | 10 | 8 | 
---------
Tree 3:
L1 | 
L6 | L8 | 
L3 | L8 | L6 | L2 | 
9 | 12 | 11 | 12 | 6 | 6 | 13 | 6 | 
---------
Tree 4:
L7 | 
L6 | L1 | 
R9 | L7 | R9 | L0 | 
14 | 10 | 8 | 11 | 3 | 16 | 10 | 3 | 
---------
Tree 5:
L0 | 
L1 | L0 | 
L0 | R5 | L0 | R2 | 
14 | 2 | 3 | 14 | 9 | 3 | 3 | 2 | 
---------
Tree 6:
R4 | 
L0 | R7 | 
R3 | R0 | R9 | L1 | 
9 | 4 | 6 | 0 | 11 | 10 | 10 | 10 | 
---------
Tree 7:
R1 | 
R0 | R1 | 
R4 | R6 | R4 | L7 | 
14 | 13 | 1 | 14 | 9 | 13 | 6 | 8 | 
---------
Tree 8:
L6 | 
R1 | R6 | 
R3 | L4 | R1 | R6 | 
11 | 10 | 8 | 8 | 10 | 8 | 10 | 13 | 
---------
Tree 9:
L4 | 
R4 | R0 | 
R3 | L3 | R4 | L9 | 
7 | 5 | 13 | 7 | 11 | 8 | 4 | 9 | 
---------
Tree 10:
L7 | 
R3 | R4 | 
L1 | R1 | L5 | R0 | 
12 | 10 | 8 | 3 | 7 | 15 | 11 | 7 | 
---------
Tree 11:
L1 | 
L2 | R1 | 
R7 | L2 | R1 | L1 | 
8 | 15 | 12 | 8 | 3 | 3 | 7 | 2 | 
---------
Tree 12:
L6 | 
L8 | R1 | 
L7 | R4 | L4 | L6 | 
13 | 15 | 13 | 8 | 10 | 8 | 7 | 7 | 
---------
Tree 13:
L6 | 
L2 | L6 | 
R7 | R0 | R0 | R7 | 
9 | 10 | 7 | 2 | 13 | 13 | 9 | 13 | 
---------
Tree 14:
L9 | 
L4 | L5 | 
L7 | R6 | L4 | R6 | 
9 | 16 | 7 | 10 | 17 | 10 | 17 | 10 | 
---------
Tree 15:
R6 | 
L2 | L2 | 
L8 | R1 | L6 | R0 | 
6 | 15 | 5 | 3 | 15 | 6 | 3 | 6 | 
---------
Tree 16:
L6 | 
L5 | L1 | 
L9 | L1 | R1 | R5 | 
6 | 17 | 10 | 8 | 12 | 1 | 8 | 6 | 
---------
Tree 17:
L6 | 
R1 | L6 | 
L3 | L9 | L1 | L6 | 
11 | 6 | 3 | 10 | 0 | 6 | 10 | 15 | 
---------
Tree 18:
R1 | 
R0 | L9 | 
L7 | R8 | L7 | L8 | 
6 | 12 | 3 | 10 | 6 | 8 | 15 | 6 | 
---------
Tree 19:
L4 | 
L4 | R8 | 
R2 | L6 | R1 | L9 | 
10 | 7 | 17 | 10 | 7 | 5 | 10 | 10 | 
---------
Tree 20:
R2 | 
R1 | L2 | 
L6 | L2 | L2 | L7 | 
8 | 15 | 3 | 3 | 3 | 4 | 4 | 3 | 
---------
Tree 21:
R0 | 
R4 | R4 | 
L6 | L8 | L1 | R2 | 
14 | 13 | 13 | 11 | 8 | 1 | 13 | 12 | 
---------
Tree 22:
R2 | 
L6 | L6 | 
L4 | R5 | L7 | R2 | 
14 | 12 | 12 | 11 | 10 | 10 | 11 | 10 | 
---------
Tree 23:
R9 | 
R6 | R9 | 
L6 | L1 | L1 | L9 | 
8 | 11 | 6 | 7 | 17 | 7 | 17 | 18 | 
---------
Tree 24:
L0 | 
L9 | L0 | 
R4 | R1 | R1 | L7 | 
4 | 7 | 9 | 10 | 9 | 10 | 9 | 7 | 
---------
Tree 25:
L6 | 
L7 | L2 | 
R9 | R7 | R4 | R8 | 
10 | 18 | 7 | 14 | 8 | 10 | 2 | 7 | 
---------
Tree 26:
R1 | 
L0 | L1 | 
L1 | L1 | L1 | R8 | 
11 | 5 | 7 | 2 | 1 | 2 | 2 | 2 | 
---------
Tree 27:
R7 | 
L6 | R0 | 
R6 | R4 | L4 | R7 | 
9 | 9 | 14 | 15 | 11 | 11 | 9 | 11 | 
---------
Tree 28:
L6 | 
L7 | L6 | 
R3 | R7 | L6 | R7 | 
14 | 8 | 12 | 14 | 11 | 10 | 10 | 13 | 
---------
Tree 29:
L4 | 
R2 | R3 | 
R8 | L0 | R0 | R0 | 
14 | 13 | 11 | 2 | 11 | 4 | 7 | 4 | 
---------
Tree 30:
L4 | 
L2 | R3 | 
R3 | R3 | R3 | R8 | 
16 | 3 | 11 | 3 | 13 | 13 | 3 | 9 | 
---------
Tree 31:
R0 | 
L4 | L3 | 
L7 | R6 | R0 | L2 | 
12 | 13 | 8 | 10 | 13 | 7 | 3 | 3 | 
---------
Tree 32:
L3 | 
R0 | R8 | 
R7 | R0 | R0 | R6 | 
4 | 16 | 5 | 5 | 5 | 3 | 11 | 9 | 
---------
Tree 33:
L0 | 
L4 | L8 | 
R4 | R2 | R1 | R6 | 
11 | 9 | 10 | 6 | 8 | 1 | 9 | 10 | 
---------
Tree 34:
R0 | 
R9 | L6 | 
R3 | L7 | L1 | L1 | 
6 | 12 | 11 | 9 | 4 | 1 | 6 | 1 | 
---------
Tree 35:
R9 | 
L1 | L7 | 
R3 | R9 | L8 | L7 | 
6 | 6 | 5 | 10 | 10 | 17 | 10 | 8 | 
---------
Tree 36:
R4 | 
R6 | L7 | 
R0 | R5 | L1 | R9 | 
9 | 4 | 15 | 11 | 12 | 4 | 15 | 18 | 
---------
Tree 37:
R9 | 
L1 | L7 | 
R3 | R2 | L1 | R9 | 
5 | 7 | 8 | 3 | 18 | 3 | 16 | 16 | 
---------
Tree 38:
L7 | 
R4 | R6 | 
R3 | L9 | R6 | L2 | 
12 | 12 | 6 | 13 | 7 | 13 | 13 | 6 | 
---------
Tree 39:
L1 | 
R7 | R0 | 
L6 | L7 | L7 | R9 | 
6 | 13 | 13 | 14 | 8 | 14 | 1 | 1 | 
---------
Tree 40:
R5 | 
R6 | R3 | 
R8 | R6 | L1 | L3 | 
10 | 13 | 8 | 9 | 9 | 6 | 3 | 6 | 
---------
Tree 41:
R9 | 
L0 | L7 | 
R8 | L0 | L7 | L0 | 
9 | 15 | 4 | 4 | 13 | 13 | 9 | 4 | 
---------
Tree 42:
R0 | 
L0 | L6 | 
L6 | R2 | R6 | R0 | 
5 | 7 | 9 | 2 | 4 | 11 | 6 | 4 | 
---------
Tree 43:
L2 | 
L7 | L7 | 
L1 | L5 | R0 | L7 | 
5 | 5 | 7 | 7 | 10 | 6 | 7 | 7 | 
---------
Tree 44:
R7 | 
R4 | R7 | 
R0 | L8 | R1 | R6 | 
4 | 8 | 13 | 8 | 5 | 6 | 11 | 16 | 
---------
Tree 45:
R1 | 
L2 | L1 | 
R6 | R1 | L5 | L0 | 
8 | 13 | 7 | 2 | 1 | 8 | 2 | 1 | 
---------
Tree 46:
R3 | 
R0 | L1 | 
R2 | L0 | L6 | R3 | 
12 | 10 | 2 | 0 | 9 | 9 | 3 | 6 | 
---------
Tree 47:
L5 | 
L4 | L5 | 
R5 | L6 | L9 | L9 | 
4 | 8 | 10 | 11 | 10 | 9 | 10 | 9 | 
---------
Tree 48:
R7 | 
L6 | R8 | 
R0 | R8 | L1 | L5 | 
5 | 5 | 6 | 14 | 10 | 8 | 12 | 9 | 
---------
Tree 49:
L6 | 
R1 | R9 | 
R7 | L0 | R6 | L1 | 
12 | 9 | 4 | 1 | 5 | 12 | 15 | 15 | 
"""


baseTrees = baseTrees.split("---------\n")
dropoutTrees = dropoutTrees.split("---------\n")
twoStageTrees = twoStageTrees.split("---------\n")
setje = set()

for treeStr in dropoutTrees:
    lines = treeStr.splitlines()
    print(lines[0])
    root = Node(lines[1].split(" | ")[0])
    root.left = Node(lines[2].split(" | ")[0])
    root.right = Node(lines[2].split(" | ")[1])
    root.left.left = Node(lines[3].split(" | ")[0])
    root.left.right = Node(lines[3].split(" | ")[1])
    root.right.left = Node(lines[3].split(" | ")[2])
    root.right.right = Node(lines[3].split(" | ")[3])
    root.left.left.left = Node(lines[4].split(" | ")[0])
    root.left.left.right = Node(lines[4].split(" | ")[1])
    root.left.right.left = Node(lines[4].split(" | ")[2])
    root.left.right.right = Node(lines[4].split(" | ")[3])
    root.right.left.left = Node(lines[4].split(" | ")[4])
    root.right.left.right = Node(lines[4].split(" | ")[5])
    root.right.right.left = Node(lines[4].split(" | ")[6])
    root.right.right.right = Node(lines[4].split(" | ")[7])
    print(root)


    if root.value.startswith("R"):
        # RIGHT FIRST IN SETJE
        if root.right.value.startswith("L"):
            if root.right.right.value == root.right.value or root.right.right.value == root.value:
                # RIGHTMOST LEAF
                if int(root.right.right.right.value) == (int(root.value[1:]) + int(root.right.value[1:])):
                    setje.add((root.value, root.right.value))
            else:
                # second to right leaf
                if int(root.right.right.left.value) == (int(root.value[1:]) + int(root.right.value[1:])):
                    setje.add((root.value, root.right.value))
        elif root.right.value == root.value:
            if root.right.right.value.startswith("L"):
                if int(root.right.right.right.value) == (int(root.value[1:]) + int(root.right.right.value[1:])):
                    setje.add((root.value, root.right.right.value))
        if root.right.left.value.startswith("L") and root.right.value != root.value:
            if int(root.right.left.right.value) == (int(root.value[1:]) + int(root.right.left.value[1:])):
                setje.add((root.value, root.right.left.value))
        if root.left.value.startswith("R") and root.left.value != root.value:
            if root.left.right.value.startswith("L"):
                if int(root.left.right.right.value) == (int(root.left.value[1:]) + int(root.left.right.value[1:])):
                    setje.add((root.left.value, root.left.right.value))
        elif root.left.value.startswith("L"):
            if root.left.right.value.startswith("R") and root.left.right.value != root.value:
                if int(root.left.right.right.value) == (int(root.left.value[1:]) + int(root.left.right.value[1:])):
                    setje.add((root.left.right.value, root.left.value))
    else:
        if root.right.value.startswith("R"):
            if root.right.right.value == root.right.value or root.right.right.value == root.value:
                # RIGHTMOST LEAF
                if int(root.right.right.right.value) == (int(root.value[1:]) + int(root.right.value[1:])):
                    setje.add((root.right.value, root.value))
            else:
                # second to right leaf
                if int(root.right.right.left.value) == (int(root.value[1:]) + int(root.right.value[1:])):
                    setje.add((root.right.value, root.value))
        elif root.right.value == root.value:
            if root.right.right.value.startswith("R"):
                if int(root.right.right.right.value) == (int(root.value[1:]) + int(root.right.right.value[1:])):
                    setje.add((root.right.right.value, root.value))
        if root.right.left.value.startswith("R") and root.right.value != root.value:
            if int(root.right.left.right.value) == (int(root.value[1:]) + int(root.right.left.value[1:])):
                setje.add((root.right.left.value, root.value))
        if root.left.value.startswith("L") and root.left.value != root.value:
            if root.left.right.value.startswith("R"):
                if int(root.left.right.right.value) == (int(root.left.value[1:]) + int(root.left.right.value[1:])):
                    setje.add((root.left.right.value, root.left.value))
        elif root.left.value.startswith("R"):
            if root.left.right.value.startswith("L") and root.left.right.value != root.value:
                if int(root.left.right.right.value) == (int(root.left.value[1:]) + int(root.left.right.value[1:])):
                    setje.add((root.left.value, root.left.right.value))
    
    if ('R3', 'L4') in setje:
        pass
            

            
    t1 = (root.value, root.right.value)
    t2 = (root.value, root.right.left.value)
    t3 = (root.left.value, root.left.right.value)

    targ = ("R7", "L6")
    targ2 = (targ[1], targ[0])


    if t1 == targ or t1 == targ2 or t2 == targ or t2 == targ2 or t3 == targ or t3 == targ2:
        print("AAAAAAAA")
    elif root.value == "R7" and root.right.right.value == "L6" and root.right.value == "R7":
        print("BBBBBBBB")
    elif root.value == "L6" and root.right.right.value == "R7" and root.right.value == "L6":
        print("CCCCCCCCC")
    #setje.add(t1)
    #setje.add(t2)
    #setje.add(t3)


output = set()
ls = list(setje)
ls.sort()
for item in ls:
    if item[0].startswith("R") and item[1].startswith("L"):
        print(item)
        output.add(item)
    if item[0].startswith("L") and item[1].startswith("R"):
        if not (item[1], item[0]) in setje:
            print(item)
            output.add(item)
    


counters = {f"R{i}": 0 for i in range(10)}
for pair in output:
    if pair[0].startswith("R"):
        counters[pair[0]] += 1
    if pair[1].startswith("R"):
        counters[pair[1]] += 1

print(counters)
# count per R digit

print(f"Total unique pairs: {len(output)}")
            
            

    
        


def count_consistent_leaves(tree_nodes):
    """
    Counts how many leaves in a depth-3 decision tree are logically consistent.
    
    :param tree_nodes: A list of exactly 15 strings representing the tree.
    :return: An integer between 0 and 8 representing the number of consistent leaves.
    """
    if len(tree_nodes) != 15:
        raise ValueError("A complete depth-3 tree must have exactly 15 nodes.")

    def count_node(index, possible_l, possible_r):
        # Base Case: We reached a leaf node
        if index >= 7:
            predicted_y = int(tree_nodes[index])
            
            # Trivially consistent if the path contains contradictions
            if not possible_l or not possible_r:
                return 1 # Changed from True
            
            # Check if the predicted class is a valid sum
            for l in possible_l:
                for r in possible_r:
                    if l + r == predicted_y:
                        return 1 # Changed from True
            
            # Inconsistent
            return 0 # Changed from False
        
        # Recursive Step: We are at a split node
        split_val = tree_nodes[index]
        digit_type = split_val[0]
        val = int(split_val[1:])
        
        # LEFT branch (Condition is FALSE -> digit != val)
        left_l, left_r = possible_l.copy(), possible_r.copy()
        if digit_type == 'L':
            left_l.discard(val)
        else:
            left_r.discard(val)
            
        # RIGHT branch (Condition is TRUE -> digit == val)
        right_l, right_r = possible_l.copy(), possible_r.copy()
        if digit_type == 'L':
            right_l = {val} if val in right_l else set()
        else:
            right_r = {val} if val in right_r else set()
            
        # Sum the consistent leaves from BOTH branches (Changed from 'and')
        return count_node(2 * index + 1, left_l, left_r) + \
               count_node(2 * index + 2, right_l, right_r)

    initial_l = set(range(10))
    initial_r = set(range(10))
    
    return count_node(0, initial_l, initial_r)


# --- String Parsing Helper ---
def parse_and_count_tree(tree_string):
    if ":" in tree_string:
        tree_string = tree_string.split(":", 1)[-1]
        
    nodes = [n.strip() for n in tree_string.replace('\n', '').split('|') if n.strip()]
    
    return count_consistent_leaves(nodes)

# --- Testing the Example ---
example_tree = """
Tree 0:
R0 | 
R4 | L4 | 
R1 | L4 | L0 | L5 | 
9 | 6 | 7 | 8 | 5 | 0 | 4 | 5 |
"""

print(f"Number of consistent leaves: {parse_and_count_tree(example_tree)} / 8")

consistentTreeCount = 0
consistentLeafCount = 0
for treeStr in baseTrees:
    count = parse_and_count_tree(treeStr)
    consistentLeafCount += count
    if count == 8:
        consistentTreeCount += 1
    else:
        pass

print(f"Total consistent leaves across all trees: {consistentLeafCount} out of {len(baseTrees) * 8}")
print(f"Number of consistent trees: {consistentTreeCount} out of {len(baseTrees)}")

def exactly_three_leaves_with_one_sum(tree_nodes):
    """
    Checks if a depth-3 decision tree has exactly 3 leaves that have 
    exactly one possible valid sum based on their constraints.
    
    :param tree_nodes: A list of exactly 15 strings representing the tree.
    :return: True if exactly 3 leaves have one possible sum, False otherwise.
    """
    if len(tree_nodes) != 15:
        raise ValueError("A complete depth-3 tree must have exactly 15 nodes.")

    def count_single_sum_leaves(index, possible_l, possible_r):
        # Base Case: We reached a leaf node
        if index >= 7:
            predicted_y = int(tree_nodes[index])
            
            # If the path is a contradiction, there are 0 possible sums
            if not possible_l or not possible_r:
                return 0
            
            # Calculate all unique sums that can be made with the remaining digits
            possible_sums = {l + r for l in possible_l for r in possible_r}
            
            # The leaf counts if there is exactly ONE unique sum possible 
            # AND the leaf correctly predicts it
            if len(possible_sums) == 1 and predicted_y in possible_sums:
                return 1
            
            return 0
        
        # Recursive Step: We are at a split node
        split_val = tree_nodes[index]
        digit_type = split_val[0]
        val = int(split_val[1:])
        
        # LEFT branch (Condition is FALSE)
        left_l, left_r = possible_l.copy(), possible_r.copy()
        if digit_type == 'L':
            left_l.discard(val)
        else:
            left_r.discard(val)
            
        # RIGHT branch (Condition is TRUE)
        right_l, right_r = possible_l.copy(), possible_r.copy()
        if digit_type == 'L':
            right_l = {val} if val in right_l else set()
        else:
            right_r = {val} if val in right_r else set()
            
        # Sum the counts from both branches
        return count_single_sum_leaves(2 * index + 1, left_l, left_r) + \
               count_single_sum_leaves(2 * index + 2, right_l, right_r)

    initial_l = set(range(10))
    initial_r = set(range(10))
    
    total_single_sum_leaves = count_single_sum_leaves(0, initial_l, initial_r)
    
    # Return True only if exactly 3 leaves meet the criteria
    return total_single_sum_leaves == 3



def parse_and_determine_optimality(tree_string):
    if ":" in tree_string:
        tree_string = tree_string.split(":", 1)[-1]
        
    nodes = [n.strip() for n in tree_string.replace('\n', '').split('|') if n.strip()]
    
    return exactly_three_leaves_with_one_sum(nodes)


optimalTreeCount = 0
for treeStr in baseTrees:
    if parse_and_determine_optimality(treeStr):
        optimalTreeCount += 1
    else:
        pass
        #lines = treeStr.splitlines()
        #print(lines[0])
        #root = Node(lines[1].split(" | ")[0])
        #root.left = Node(lines[2].split(" | ")[0])
        #root.right = Node(lines[2].split(" | ")[1])
        #root.left.left = Node(lines[3].split(" | ")[0])
        #root.left.right = Node(lines[3].split(" | ")[1])
        #root.right.left = Node(lines[3].split(" | ")[2])
        #root.right.right = Node(lines[3].split(" | ")[3])
        #root.left.left.left = Node(lines[4].split(" | ")[0])
        #root.left.left.right = Node(lines[4].split(" | ")[1])
        #root.left.right.left = Node(lines[4].split(" | ")[2])
        #root.left.right.right = Node(lines[4].split(" | ")[3])
        #root.right.left.left = Node(lines[4].split(" | ")[4])
        #root.right.left.right = Node(lines[4].split(" | ")[5])
        #root.right.right.left = Node(lines[4].split(" | ")[6])
        #root.right.right.right = Node(lines[4].split(" | ")[7])
        #print(root)


print(f"Number of optimal trees: {optimalTreeCount} out of {len(baseTrees)}")

# modify x-ticks such that all even numbers are shown
import matplotlib.pyplot as plt

# plot counts of number of times each class is left-most leaf
counts = {i: 0 for i in range(19)}
for treeStr in baseTrees:
    lines = treeStr.splitlines()
    counts[int(lines[4].split(" | ")[0])] += 1
s = sum(counts.values())
for key in counts:
    counts[key] /= s
plt.bar(counts.keys(), counts.values())
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.title("Distribution of Left-Most Leaves for Standard CFR")
plt.xticks(range(0, 20, 2))
plt.show()

counts = {i: 0 for i in range(19)}
for treeStr in dropoutTrees:
    lines = treeStr.splitlines()
    counts[int(lines[4].split(" | ")[0])] += 1
s = sum(counts.values())
for key in counts:
    counts[key] /= s
plt.bar(counts.keys(), counts.values())
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.title("Distribution of Left-Most Leaves for Dropout CFR")
plt.xticks(range(0, 20, 2))
plt.show()

counts = {i: 0 for i in range(19)}
for treeStr in twoStageTrees:
    lines = treeStr.splitlines()
    counts[int(lines[4].split(" | ")[0])] += 1
s = sum(counts.values())
for key in counts:
    counts[key] /= s
plt.bar(counts.keys(), counts.values())
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.title("Distribution of Left-Most Leaves for Two-Stage CFR")
plt.xticks(range(0, 20, 2))
plt.show()