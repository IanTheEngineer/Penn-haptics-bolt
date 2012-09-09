% Naomi Fitter
% Code for creating dendrogram from Correspondence Analysis test results

%%
clear
clc
clf

%%
[eig,labels] = xlsread('AbsDist.xlsx','Sheet2');
[eig2,labels2] = xlsread('AbsDist.xlsx','Sheet1');

label_cols = ['cool        '; 'springy     '; 'hard        '; 'porous      '; 'rough       '; 'thin        '; 'slippery    ';...
    'compressible'; 'hollow      '; 'smooth      '; 'compact     '; 'bumpy       '; 'squishy     '; 'sticky      ';...
    'unpleasant  '; 'gritty      '; 'plasticky   '; 'soft        '; 'fuzzy       '; 'solid       '; 'thick       '; 'stiff       ';...
    'deformable  '; 'nice        '; 'fibrous     '; 'textured    '; 'elastic     '; 'meshy       '; 'hairy       ';...
    'absorbent   '; 'grainy      '; 'crinkly     '; 'scratchy    '; 'metallic    '];

label_rows = ['101'; '103'; '104'; '105'; '106'; '108'; '109'; '110'; '111'; '113';...
    '114'; '115'; '118'; '119'; '120'; '201'; '202'; '206'; '208'; '211'; '212';...
    '213'; '214'; '216'; '218'; '219'; '301'; '303'; '309'; '313'; '314'; '315';...
    '316'; '317'; '318'; '319'; '320'; '401'; '403'; '406'; '407'; '408'; '410';...
    '501'; '502'; '503'; '504'; '508'; '509'; '601'; '602'; '701'; '702'; '703'];

word_label_rows = ['Blue Car Sponge       ';'Yellow Soft Foam      ';'Gray Stiff Foam       ';
    'Pool Noodle           ';'Shelf Liner           ';'Pink Stiff Foam       ';'Orange Car Sponge     ';
    'Black Rough Foam      ';'Applicator Pad        ';'Marker Eraser         ';'Kitchen Sponge        ';
    'Koozie                ';'Gray Bumpy Foam       ';'Gray Smooth Foam      ';'Black Smooth Foam     ';
    'Soap Dispenser        ';'Tarp                  ';'Index Card Case       ';'Bath Cloth            ';
    'Black Acrylic         ';'Smooth Yellow Acrylic ';'Cutting Board         ';'Bubble Wrap           ';
    'Plastic Case          ';'Plastic Dispenser     ';'Rough Yellow Acrylic  ';'Pen Case              ';
    'Hardcover Book        ';'Toilet Paper          ';'Kleenex Pack          ';'Blue Toothpaste Box   ';
    'Cookie Box            ';'Red Toothpaste Box    ';'Cosmetics Box         ';'Cushioned Envelope    ';
    'Notepad               ';'Fiberboard            ';'Satin Pillowcase      ';'Placemat              ';
    'Soft Chalkboard Eraser';'Dishcloth             ';'Hard Chalkboard Eraser';'Cloth Sack            ';
    'Corkboard             ';'Coco Liner            ';'Loofah                ';'Concrete              ';
    'Brick                 ';'Silicone Block        ';'Glass Bottle          ';'Glass Container       ';
    'Metal Channel         ';'Metal Vase            ';'Metal Block           '];


%%
c = cellstr(label_cols);
distances = pdist(eig);
Z = linkage(distances);
[H,T] = dendrogram(Z,0,'labels',c,'colorthreshold',1.5,'orientation','right');

%%
figure(2)
d = cellstr(word_label_rows);
distances2 = pdist(eig2);
Z2 = linkage(distances2);
[H2,T2] = dendrogram(Z2,0,'labels',d,'colorthreshold',1.5,'orientation','right');
