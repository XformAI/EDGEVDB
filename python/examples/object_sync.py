#!/usr/bin/env python3
"""
Example: Object store with relations and sync.

Demonstrates:
  - Creating typed objects (Document, Author, Tag)
  - Adding relations between objects
  - Basic sync delta export
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    from edgevdb import EdgeVDB

    with EdgeVDB("./edgevdb_objects") as db:
        # Create authors
        alice_id = db.put_object("Author", {"name": "Alice", "affiliation": "MIT"})
        bob_id = db.put_object("Author", {"name": "Bob", "affiliation": "Stanford"})
        print(f"Authors: Alice={alice_id}, Bob={bob_id}")

        # Create documents
        doc1_id = db.put_object("Document", {
            "title": "Introduction to Vector Databases",
            "year": 2024,
            "abstract": "Overview of HNSW and product quantization"
        })
        doc2_id = db.put_object("Document", {
            "title": "On-Device RAG Systems",
            "year": 2025,
            "abstract": "Retrieval-augmented generation on mobile"
        })
        print(f"Documents: doc1={doc1_id}, doc2={doc2_id}")

        # Create tags
        ai_tag = db.put_object("Tag", {"name": "AI"})
        db_tag = db.put_object("Tag", {"name": "Database"})

        # Add relations
        db.add_relation("authored_by", doc1_id, alice_id)
        db.add_relation("authored_by", doc1_id, bob_id)
        db.add_relation("authored_by", doc2_id, alice_id)
        db.add_relation("tagged_with", doc1_id, ai_tag)
        db.add_relation("tagged_with", doc1_id, db_tag)
        db.add_relation("tagged_with", doc2_id, ai_tag)
        print("Relations created")

        # Query
        alice = db.get_object(alice_id)
        print(f"\nAlice: {alice}")

        doc1 = db.get_object(doc1_id)
        print(f"Doc 1: {doc1}")

        print("\nDone! Objects saved to ./edgevdb_objects/")


if __name__ == "__main__":
    main()
