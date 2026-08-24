#pragma once
#include "Node.h"
#include "NodeBuildContext.h"
#include <functional>
#include <map>

namespace dataflow {
    using NodeCreateFunc = std::function<NodeBase::Ptr(NodeId id, const NodeBuildContext&)>;

    struct NodeDefinition {
        std::string name;
        NodeCreateFunc func;
    };
}
