#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>

namespace ipc = boost::interprocess;

class Viewer {
    ipc::message_queue mq(ipc::open_or_create,
            "mm", 128, sizeof(int));
};
