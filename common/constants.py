class BasicClass(object):
    LOG_LEVEL_ENV = "AUTO_RL_LOG_LEVEL"

class CommunicationReqMeta(object):
    COMM_META_JOB_UID = "job_uid"
    COMM_META_JOB_UID_INVALID_MSG = "Job uid is invalid"

class NodeType(object):
    TRAIN_NODE = "TRAIN_NODE"
    INFER_NODE = "INFER_NODE"
    COLOC_NODE = "COLOC_NODE"
    TEMP_NODE = "TEMP_NODE"