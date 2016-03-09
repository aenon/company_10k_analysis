
# coding: utf-8

# In[81]:

import pandas as pd
import os


# In[52]:

Labels = pd.read_csv("../LabellingMatrix.csv")


# In[88]:

def rename_POS_NEG( FOLDER ):
    try:
        os.chdir(FOLDER)
    except:
        print "Error pushing into ", FOLDER
        return "Error"
    else:
        print "In ", FOLDER
    
    FILES = os.listdir(".")
    if len(FILES) == 0:
        print "No files"
        os.chdir("/data/Mar08_2042_SP500_MDNAs_Label/")
        return "Error"
    
    YEARS = range(1990, 2016)
    
    for YEAR in YEARS:
        for FILE in FILES:
            if FILE.startswith("MDNA_"+str(YEAR)):
                if list(Labels[FOLDER][Labels["Year"]==YEAR] == "POS")[0]:
                    print "Renaming", FILE
                    os.rename(FILE, FILE + ".POS")
                elif list(Labels[FOLDER][Labels["Year"]==YEAR] == "NEG")[0]:
                    print "Renaming", FILE
                    os.rename(FILE, FILE + ".NEG")
    print "Done renaming files in ", FOLDER
    os.chdir("/data/Mar08_2042_SP500_MDNAs_Label/")


# In[89]:

os.chdir("/data/Mar08_2042_SP500_MDNAs_Label/")


# In[90]:

FOLDERS = os.listdir(".")


# In[91]:

for FOLDER in FOLDERS:
    print FOLDER


# In[92]:

for FOLDER in FOLDERS:
    try:
        rename_POS_NEG(FOLDER)
    except:
        print "Error trying folder ", FOLDER
    else:
        print "Folder", FOLDER, "renamed"


# In[71]:

rename_POS_NEG(FOLDERS[6])


# In[6]:

FOLDER = FOLDERS[3]


# In[7]:

pushd "$FOLDER"


# In[10]:

FILES = os.listdir(".")


# In[39]:

YEARS = range(1990, 2016)


# In[40]:

for YEAR in YEARS:
    for FILE in FILES:
        if FILE.startswith("MDNA_"+str(YEAR)):
            if list(Labels[FOLDER][Labels["Year"]==YEAR] == "POS")[0]:
                os.rename(FILE, FILE + ".POS")
            elif list(Labels[FOLDER][Labels["Year"]==YEAR] == "NEG")[0]:
                os.rename(FILE, FILE + ".NEG")


# In[41]:

ls


# In[21]:

FILE


# In[18]:

YEARS = range(1990, 2016)


# In[22]:

YEAR = YEARS[13]


# In[23]:

YEAR


# In[24]:

if Labels[FOLDER][Labels["Year"]==YEAR] == "POS":
    print FILE, "is POS"
elif Labels[FOLDER][Labels["Year"]==YEAR] == "NEG":
    print FILE, "is NEG"


# In[25]:

Labels[FOLDER]


# In[26]:

Labels["Year"]==YEAR


# In[37]:

if list(Labels[FOLDER][Labels["Year"]==YEAR] == "POS")[0]:
    print "POS"


# In[ ]:



