from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="dashboard-index"),
    path("stations3d/", views.stations3d_view, name="dashboard-stations3d"),
    path("predict/", views.predict_view, name="dashboard-predict"),
    path("anomalies/", views.anomalies_view, name="dashboard-anomalies"),
    path("anomalies/upload/", views.upload_data_view, name="dashboard-upload"),
    path("anomalies/<str:machine_id>/", views.machine_detail_view, name="dashboard-anomaly-detail"),
    path("compare/", views.compare_view, name="dashboard-compare"),
    path("maintenance/", views.maintenance_view, name="dashboard-maintenance"),
]
