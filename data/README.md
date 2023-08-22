### Folders Description

  - `AVEC_16_data`: This folder contains the **DAIC-WOZ** dataset ("Distress Analysis Interview
Corpus - Wizard of Oz").
  - `AVEC_19_data`: This folder contains the **E-DAIC** dataset ("Extended Distress Analysis Interview Corpus").

**NOTE:** Due to license restriction of the datasets, inside each folder we have only uploaded files with the expected name and format with some fake examples. Please **download the original datasets from [https://dcapswoz.ict.usc.edu/](https://dcapswoz.ict.usc.edu/)**. Converted to the same tab-separated format shown below, and replace the files that are inside these folders.

Each dataset file is expected to be named `SPLIT_all_data.txt` (e.g. `test_all_data.txt`) and contains one interview per line along with the ground truth label separated by tab character, as in the following example:

```
positive	<synch> <laughter> yes i'm doing well i was born in...
negative	<sync> yes i'm okay i was born in ...
...
```

Please see the example files inside each folder.
