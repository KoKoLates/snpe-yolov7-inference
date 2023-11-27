#include <glib.h>
#include <gst/gst.h>

static void
usage(const char *argv) {
    g_print("Usage: %s [camera index] [ip] [port].\n", argv);
    g_print("IP: x.x.x.x: IP address of ground station. It maybe through eth0 or wlan0.\n");
    g_print("Examle:\n");
    g_print("   %s 0 192.168.8.10 11024\n", argv);
}

static gboolean 
bus_callback(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *)data;
    int type = GST_MESSAGE_TYPE(msg);

    if (type == GST_MESSAGE_ERROR) {
        gchar   *debug;
        GError  *error;
        gst_message_parse_error(msg, &error, &debug);
        g_free(debug);
        g_printerr("Error: %s\n.", error->message);
        g_error_free(error);
        g_main_loop_quit(loop);
    }
    return TRUE;
}

int
main(int argc, char **argv) {
    GstBus *bus;
    GMainLoop *loop;
    GstElement *pipeline, *source, *framefilter, *parser, *queue, *payloader, *sink;

    gint port;
    guint bus_idx;
    gint8 camera_idx;

    /* initialize gstremaer */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    if (argc != 4) {
        usage(argv[0]);
        return -1;
    }

    camera_idx = atoi(argv[1]);
    if (camera_idx < 0 || camera_idx > 3) {
        g_printerr("camera index incorrect.\n");
        usage(argv[0]);
        return -1;
    }

    port = atoi(argv[3]);
    if (port < 0) {
        g_print("port format incorrect.\n");
        usage(argv[0]);
        return -1;
    }

    /* create elements */
    pipeline    = gst_pipeline_new("video-streaming");
    source      = gst_element_factory_make("qtiqmmfsrc",    "qmmf-source");
    framefilter = gst_element_factory_make("capsfilter",    "frame-filter");
    parser      = gst_element_factory_make("h264parse",     "h264-parser");
    queue       = gst_element_factory_make("queue",         "queue");
    payloader   = gst_element_factory_make("rtph264pay",    "rtp-payloader");
    sink        = gst_element_factory_make("udpsink",       "udp-sink");

    if (!pipeline || !source || !framefilter || !parser || ! queue || !payloader || !sink) {
        g_print("Create pipeline elements failed.\n");
        return -1;
    }

    g_object_set(G_OBJECT(source), "camera", camera_idx, NULL);
    if (camera_idx == 0) {
        g_object_set(G_OBJECT(framefilter), "caps", 
            gst_caps_from_string("video/x-h264, framerate=30/1, width=1920, height=1080"), NULL);
    } else {
        g_object_set(G_OBJECT(framefilter), "caps", 
            gst_caps_from_string("video/x-h264, framerate=30/1, width=1280, height=720"),  NULL);
    }
    g_object_set(G_OBJECT(parser), "config-interval", 1, NULL);
    g_object_set(G_OBJECT(sink), "host", argv[2], "port", port, NULL);

    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_idx = gst_bus_add_watch(bus, bus_callback, loop);
    gst_object_unref(bus);

    gst_bin_add_many(GST_BIN(pipeline), source, framefilter, parser, queue, payloader, sink, NULL);
    if(!gst_element_link_many(source, framefilter, parser, queue, payloader, sink, NULL)) {
        g_printerr("Elements could not be linked.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    g_print("Start -> %s:%d", argv[2], port);
    g_main_loop_run(loop);

    g_print("Stop\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_idx);
    g_main_loop_unref(loop);
    
    gst_deinit();
    return 0;
}