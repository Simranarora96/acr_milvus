from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="video_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2048)
]

schema = CollectionSchema(fields, description="Video Embedding Collection")
collection = Collection("video_fingerprints", schema)
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
collection.load()
print("✅ Milvus collection ready.")
