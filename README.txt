USAGE OF THE STWR CLASSIFIER
*************************************
Master Thesis
author: Luise Schricker

Dependencies and package-versions are
noted in the respective Python-files.
*************************************

PREPARATION:
- general:
1.) You need to get the list of reporting words compiled by Annelen Brunner and put it into the directory "data/stw_words". You can find this list in the additional material to her dissertation: https://repos.ids-mannheim.de/tou.html (tools/rule_based_tools/MarkSTWWords/stw_words/stw_words.xls)
2.) You need to download the RFTagger and put it into the directory "RFTagger" (directories in this folder should be bin/cmd/doc etc.): http://www.cis.uni-muenchen.de/∼schmid/tools/RFTagger/
3.) OPTIONAL: If you want to use the full candidate speaker features, you need to get a list of possible synonyms for the word "Person", e.g. by extracting the hyponyms from Germanet and put these word (one per line) into the empty text file "data/person.txt"

- for training:
1.) Get the STWR corpus and put it into data/corpus: https://repos.ids-mannheim.de/tou.html



Command Line Usage:
NOTE: All processes can take some time.

- Annotation:
python3 STWR_recognition.py <path_to_file_to_be_annotated>

Optional parameters:
--html      Indicates that an html visualization of the annotation should be produced. (Flag)
            If the output-parameter is not given, the html visualization will be saved as "text_annotated.html".
--post      Indicates that the span postprocessing step should be executed for the annotations. (Flag)
--output    Name of the output file (without extension). If this parameter is not given,
            the annotated text will be saved as "text_annotated.tsv".


- Training:
python3 STWR_recognition.py <path_to_corpus_data> --train

Optional parameters:
--ml        Type of ML to be used for training. One of 'random_forest', 'svm', 'neural'.
--augment   Type of data augmentation to be used for training. One of 'oversampling', 'SMOTE', 'augmentation'.
--no_sequ   Indicates that the system should be retrained without sequential label features. (Flag)


- Evaluation:
python3 STWR_recognition.py <path_to_corpus_data> --eval

Optional parameters:
-- post   Indicates that the span postprocessing step should be executed for the annotations. (Flag)



Example Usages:

Annotation: python3 STWR_recognition.py text.txt --html --output text_annotated
Training:   python3 STWR_recognition.py corpusExtracted --train --ml svm
Evaluation: python3 STWR_recognition.py corpusExtracted --eval --post