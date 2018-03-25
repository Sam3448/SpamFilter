from elasticsearch import Elasticsearch
import os
import fileinput

#**********************************Index**************************************

def index_document(es, docIndex, docType, doc_id, doc_text, doc_label):
    status = ''
    if not es.exists(index=docIndex, doc_type=docType, id=doc_id):
        es.index(index=docIndex, doc_type=docType, id=doc_id, body={"content": doc_text, "label" : doc_label})
        status = 'CREATING'
    else:
        es.update(index=docIndex, doc_type=docType, id=doc_id, body={docType: {"content": doc_text, "label" : doc_label}})
        status = 'UPDATING'

    return status

def index(docHeader, datadirs, docIndex, docType):
    es = Elasticsearch()
    if not es.indices.exists(index = docIndex):
        es.indices.create(index=docIndex)

    mapping = {
      "properties": {
        "content": {
          "type": "text",
          "analyzer": "standard",
          "search_analyzer": "standard"
        },
        "label": {
            "type": "text"
        }
      }
    }

    es.indices.put_mapping(index=docIndex, doc_type=docType, body=mapping)
    print(es.indices.get_mapping(index=docIndex, doc_type=docType))

    totalcount = 0
    indexLen = len(docIndex)

    for datadir in datadirs:
        sentenceCount = 0
        subdatadir = datadir[: datadir.rfind('/')]
        header = subdatadir[subdatadir.rfind('/') + 1 : ]
        for filename in os.listdir(datadir):#read in every training file
            if not filename.startswith(docHeader):
                continue

            print(filename)
            for line in fileinput.input(datadir + "/" + filename):
                doc_id = header + str(sentenceCount)
                sentenceCount += 1
                doc_text = line
                status = index_document(es, docIndex, docType, doc_id, doc_text, header)
                totalcount += 1
                print("%s ==> current file Number : %s ; %d " % (status, doc_id, totalcount))


#**********************************Search**************************************

def search(query, docIndex, topK):
    es = Elasticsearch()

    queryStr = '''{
        "size" : %s,
        "query": {
            "query_string" : {
                "query" : \"%s\"
            }
        }
    }'''%(str(topK), query)

    response = es.search(index=docIndex, body = queryStr)
    return response

#**********************************Util**************************************

def KNN(response, posTag, negTag):
    if int(response['hits']['total']) == 0:
        # print("NO HITS")
        return posTag # can't reduce noise, keep result the same

    posScore, negScore = 0, 0
    for each_doc in response['hits']['hits']:
        curLabel = each_doc['_source']['label']
        curScore = each_doc['_score']
        if curLabel == posTag:
            posScore += curScore
        elif curLabel == negTag:
            negScore += curScore

    # print("pos score : %.3f, neg score : %.3f \n" % (posScore, negScore))
    return posTag if posScore >= negScore else negTag


