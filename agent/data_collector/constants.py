class CollectedDataType:
    STACK_TRACE = "STACK_TRACE"
    TRAINING_LOG = "TRAINING_LOG"
    TRAINING_METRIC = "TRAINING_METRIC"
    RESOURCE_METRIC = "RESOURCE_METRIC"
    DIAGNOSIS_OBSERVATION = "DIAGNOSIS_OBSERVATION"
    GENERIC = "GENERIC"

class CollectedNodeType:
    TRAIN_NODE = "TRAIN_NODE"
    INFER_NODE = "INFER_NODE"
    COLOC_NODE = "COLOC_NODE"

class DiagnosisErrorConstant:
    """Error constants for diagnosis"""
    GPU_LOST = "GPU is lost"
    PRE_CHECK_FAILED = "Pre-check failed"
    NODE_FAILED = "node_failed"
    RESOURCE_COLLECT_ERROR = "resource_collect_error"