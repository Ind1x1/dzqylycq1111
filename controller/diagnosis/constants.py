class DiagnosisActionType(object):
    # common
    NONE = "no_action"
    ANY = "any_action"
    LOG = "log"

    # master operation
    JOB_ABORT = "job_abortion"
    MASTER_RELAUNCH_WORKER = "master_relaunch_worker"
    EVENT = "event"

    # node operation
    RESTART_WORKER = "restart_worker"
    RELAUNCH_WORKER = "relaunch_worker"

class DiagnosisConstant(object):
    MASTER_DIAGNOSIS_OBSERVING_INTERVAL_SECS = 180
    METRIC_COLLECT_INTERVAL_SECS = 60
    CHECK_TENSOR_DEFAULT_RECORDS = 30

    AGENT_PERIODICALLY_REPORT_INTERVAL_SECS = 15
    MASTER_INSTANCE = -1
    ANY_INSTANCE = -2
    LOCAL_INSTANCE = -3
    ACTION_EXPIRED_TIME_PERIOD_DEFAULT = 60 * 5
    MAX_ACTION_QUEUE_SIZE = 1000

    MIN_DIAGNOSIS_INTERVAL = 15

class DiagnosisDataType:
    STACK_TRACE = "STACK_TRACE"
    TRAINING_LOG = "TRAINING_LOG"
    TRAINING_METRIC = "TRAINING_METRIC"
    RESOURCE_METRIC = "RESOURCE_METRIC"
    GENERIC = "GENERIC"