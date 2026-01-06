data info:
Ash policy data is as on 1/1/2026
rest of the info othet than ash info are older as on first version 

code files:;
app.py: this is the main app to query from the cloud db
app2.py: this file has option to upload the docs to upload info into the db


info and important commands:
#Vector storage ≈ 1440 × 1.5 KB ≈ 2.1 MB.
#always add in pdf format
#important commands/instructions/steps to smooth run and fast. 
#curl -X DELETE "http://15.206.197.214:6333/collections/rag_docs"
#above command to remove content from the db

db info:
upload documents using app2.py
uploading documents with normal text and in pdf format will be extracted more info, it can able to learn in normal text only
once you uploaded thet documents, just move them into dummy docs for reference. because if you want to upload one more document with existing document is in that folder only, if you build index it will again learn two docuemtns( means whatever the documents are in the original doc folder)
if you remove documents after building, the db info will not vanish. only document will be vanished.


