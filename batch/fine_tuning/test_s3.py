import s3fs
#fs = s3fs.S3FileSystem(anon=True,key="AKIA5FUF7WFNKLJ5XF7L", secret="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")
fs = s3fs.S3FileSystem(anon=True)

fs.ls('graphcore-ce/andyw')