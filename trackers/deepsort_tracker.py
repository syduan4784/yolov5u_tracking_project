from deep_sort_realtime.deepsort_tracker import DeepSort

def create_tracker():
    """Khởi tạo DeepSORT tracker"""
    tracker = DeepSort(
        max_age=30,
        n_init=2,
        nms_max_overlap=1.0
    )
    return tracker
