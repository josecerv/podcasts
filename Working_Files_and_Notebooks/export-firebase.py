#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import firebase_admin
from firebase_admin import credentials, firestore
import json
import datetime

cred = credentials.Certificate('podcast-4c7aa-firebase-adminsdk-nj0md-8a8359518a.json')
firebase_admin.initialize_app(cred)

# Check if the app is already initialized
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

def export_collection(collection_name):
    try:
        docs = db.collection(collection_name).stream()
        data = []
        for doc in docs:
            doc_dict = doc.to_dict()
            doc_dict['id'] = doc.id
            # Convert DatetimeWithNanoseconds to ISO format strings
            for key, value in doc_dict.items():
                if isinstance(value, datetime.datetime):
                    # Convert to ISO format string
                    doc_dict[key] = value.isoformat()
            data.append(doc_dict)
        with open(f'{collection_name}.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Exported {len(data)} documents from {collection_name} collection')
    except Exception as e:
        print(f'An error occurred: {e}')

# Replace 'survey_responses' with your collection name
export_collection('survey_responses')
export_collection('podcast_clicks')

