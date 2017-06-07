#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/quantized_layer.hpp"
#include "caffe/util/format.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const Net* root_net)
    : root_net_(root_net), solver_(NULL) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase, const Net* root_net)
    : root_net_(root_net), solver_(NULL) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  param.mutable_state()->set_phase(phase);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  CHECK(Caffe::root_solver() || root_net_)
      << "root_net_ needs to be set for all non-root solvers";
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  if (phase_ == TRAIN) {
    caffe::P2PSync<Dtype>::divide_batch_size(&filtered_param);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  memory_used_ = 0;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // For non-root solvers, whether this layer is shared from root_net_.
    bool share_from_root = !Caffe::root_solver()
        && root_net_->layers_[layer_id]->ShareInParallel();
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    if (share_from_root) {
      LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
      layers_.push_back(root_net_->layers_[layer_id]);
      layers_[layer_id]->SetShared(true);
    } else {
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    }
    layer_names_.push_back(layer_param.name());
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
    bool need_backward = false;

    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    if (share_from_root) {
      // Set up size of top blobs using root_net_
      const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
      const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];
      for (int top_id = 0; top_id < base_top.size(); ++top_id) {
        this_top[top_id]->ReshapeLike(*base_top[top_id]);
        LOG(INFO) << "Created top blob " << top_id << " (shape: "
            << this_top[top_id]->shape_string() <<  ") for shared layer "
            << layer_param.name();
      }
    } else {
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() != 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();

  // invert param_layer_indices_ to give map of
  // (level_id, local param_id) -> global param_id
  for (int i = 0; i < param_layer_indices_.size(); ++i) {
    layer_index_params_[param_layer_indices_[i]] = i;
  }

  debug_info_ = param.debug_info();
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  available_blobs->erase(blob_name);
  bool need_backward = blob_need_backward_[blob_id];
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);
  }
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      if (debug_info_) { BackwardDebugInfo(i); }

      // reduce gradients as soon as they are ready
      if (Caffe::solver_count() > 1) {
#ifndef CPU_ONLY
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
#endif
        for (int j = 0; j < layers_[i]->blobs().size(); ++j) {
          int param_id = layer_index_params_[make_pair(i, j)];

          // check if we need to synchronize after reduction
          bool need_sync = false;
          // If param has been split, update owner and sync
          if (param_owners_[param_id] >= 0) {
            param_id = param_owners_[param_id];
            need_sync = true;
          }

          for (int k = 0; k < solver_->callbacks().size(); ++k) {
            solver_->callbacks()[k]->allreduce(param_id);
          }

          // perform synchronization if needed
          if (need_sync) {
            for (int k = 0; k < solver_->callbacks().size(); ++k) {
              solver_->callbacks()[k]->syncCommStream();
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      if(target_blobs[j]->count() != source_blob->count()) {
		  CHECK(target_blobs[j]->shape() == source_blob->shape())
			  << "Cannot share param " << j << " weights from layer '"
			  << source_layer_name << "'; shape mismatch.  Source param shape is "
			  << source_blob->shape_string() << "; target param shape is "
			  << target_blobs[j]->shape_string();
      } else {
		  if(target_blobs[j]->shape() != source_blob->shape()) {
			  LOG(WARNING)  << "Shape mismatch, param: " << j << " layer: "
				  << source_layer_name << " source: "
				  << source_blob->shape_string() << " target: "
				  << target_blobs[j]->shape_string();
		  }
      }
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    if(target_blobs.size()!= source_layer.blobs_size()) {
        LOG(WARNING) << "Incompatible number of blobs for layer " << source_layer_name;
    }
    int num_blobs_to_copy = std::min<int>(target_blobs.size(), source_layer.blobs_size());
    for (int j = 0; j < num_blobs_to_copy; ++j) {
	  bool do_copy = true;
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        LOG(WARNING) << "Copying from " << source_layer_name << " to " <<
            layers_[target_layer_id]->layer_param().name() <<
            " target blob " << j;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        if(target_blobs[j]->count() != source_blob.count()) {
        	do_copy = false;
            LOG(WARNING) << "Cannot copy param " << j << " weights from layer '"
                << source_layer_name << "'; shape mismatch.  Source param shape is "
                << source_blob.shape_string() << "; target param shape is "
                << target_blobs[j]->shape_string() << ". "
                << "To learn this layer's parameters from scratch rather than "
                << "copying from a saved net, rename the layer.";
        } else {
            LOG(WARNING) << "Shape mismatch, param: " << j << " layer: "
                << source_layer_name << " source: "
                << source_blob.shape_string() << " target: "
                << target_blobs[j]->shape_string() << ". ";
        }
      }
      const bool ignore_mismatching_blobs = ((solver_==NULL) || solver_->param().ignore_mismatching_blobs());
      if(do_copy) {
          const bool kReshape = false;
          target_blobs[j]->FromProto(source_layer.blobs(j), kReshape, ignore_mismatching_blobs);
      } else if(!ignore_mismatching_blobs) {
          LOG(WARNING) << "ignore_mismatching_blobs: " << ignore_mismatching_blobs;
          LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
              << source_layer_name << "'; shape mismatch.";
      }
    }
  }
  CopyQuantizationRangeInLayers();
}

template <typename Dtype>
int Net<Dtype>::GetSparsity(std::map<std::string, std::pair<int,int> >& sparsity_map, const Dtype threshold){
  int blob_count = 0;
  sparsity_map.clear();
  int max_params_to_check = 1;
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {	
      const LayerParameter& layer_param = layers_[layer_id]->layer_param();  
      if(layer_param.type() == "Convolution" || layer_param.type() == "InnerProduct") {
          int num_params_to_check = std::min<int>(max_params_to_check, layers_[layer_id]->blobs().size());
		  for (int param_id = 0; param_id < num_params_to_check;++param_id) {
		    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
		    const int net_param_id = param_id_vecs_[layer_id][param_id];
		    const string& blob_name = param_display_names_[net_param_id];
		    //const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
		    std::pair<int,int> sp_map = std::make_pair(blob.count_zero(threshold), blob.count());
		    sparsity_map[layer_names_[layer_id] + "_param_" + blob_name] = sp_map;
			blob_count++;
		  }	
	  }
  }
  return blob_count;
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff, bool write_blobs) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff, write_blobs);
  }
}


template<typename Dtype>
void Net<Dtype>::ToProtoLog(NetLogParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO)<< "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    // Setting write_blobs to false as blobs will be written separately below
    layers_[i]->ToProto(layer_param, write_diff, false);

    if (layers_[i]->layer_param().snapshot_log_weights()) {
      const vector<shared_ptr<Blob<Dtype> > >& this_blobs = layers_[i]->blobs();
      for (int blob_id = 0; blob_id < this_blobs.size(); blob_id++) {
        BlobProtoLog* layer_blob_log = param->add_blob_log();
        BlobProto *blob_data = layer_blob_log->mutable_blob_data();
        Blob<Dtype> blob_copy;
        blob_copy.CopyFrom(*this_blobs[blob_id], false, true);
        string blob_name = layer_names_[i] + " weight:" + caffe::format_int(blob_id);
        layer_blob_log->set_name(blob_name);
        if (layers_[i]->layer_param().has_quantization_param()) {
          const QuantizationParameter& qparam = layers_[i]->layer_param().quantization_param();
          if (qparam.quantize_layer_weights()) {
            blob_name += " quantized";
            layer_blob_log->set_is_quantized(true);
            layer_blob_log->set_is_unsigned(false);
            layer_blob_log->set_bw(qparam.bw_weights());
            layer_blob_log->set_fl(qparam.fl_weights());
            Convert2FixedPoint_cpu(blob_copy.mutable_cpu_data(), blob_copy.count(), qparam.bw_weights(),
                qparam.fl_weights(), false, (blob_id == 0));
          }
        }
        blob_copy.ToProto(blob_data, false);
      }
    }

    if (layers_[i]->layer_param().snapshot_log_in()) {
      const vector<Blob<Dtype>*>& this_bottom = this->bottom_vecs_[i];
      int num_bottom_blobs = this_bottom.size();
      for (int bottom_id = 0; bottom_id < num_bottom_blobs; bottom_id++) {
        BlobProtoLog* layer_blob_log = param->add_blob_log();
        BlobProto *blob_data = layer_blob_log->mutable_blob_data();
        Blob<Dtype> blob_copy;
        blob_copy.CopyFrom(*this_bottom[bottom_id], false, true);
        std::stringstream ss;
        ss << "layer:" << layer_names_[i] << ", bottom:" << bottom_id;
        string blob_name = ss.str();
        layer_blob_log->set_name(blob_name);
        if (layers_[i]->layer_param().has_quantization_param()) {
          const QuantizationParameter& qparam = layers_[i]->layer_param().quantization_param();
          if (qparam.quantize_layer_in()) {
            blob_name += " : quantized";
            layer_blob_log->set_is_quantized(true);
            bool is_unsigned = qparam.unsigned_layer_in_size()>0 && qparam.unsigned_layer_in(bottom_id);
            layer_blob_log->set_is_unsigned(is_unsigned);
            layer_blob_log->set_bw(qparam.bw_layer_in());
            if(qparam.fl_layer_in_size() > bottom_id) {
              layer_blob_log->set_fl(qparam.fl_layer_in(bottom_id));
              Convert2FixedPoint_cpu(blob_copy.mutable_cpu_data(), blob_copy.count(), qparam.bw_layer_in(),
                  qparam.fl_layer_in(bottom_id), is_unsigned, true);
            }
          }
        }
        blob_copy.ToProto(blob_data, false);
      }
    }

    if (layers_[i]->layer_param().snapshot_log_out()) {
      const vector<Blob<Dtype>*>& this_top = this->top_vecs_[i];
      for (int top_id = 0; top_id < this_top.size(); top_id++) {
        BlobProtoLog* layer_blob_log = param->add_blob_log();
        BlobProto *blob_data = layer_blob_log->mutable_blob_data();
        Blob<Dtype> blob_copy;
        blob_copy.CopyFrom(*this_top[top_id], false, true);
        std::stringstream ss;
        ss << "layer:" << layer_names_[i] << ", top:" << top_id;
        string blob_name = ss.str();
        layer_blob_log->set_name(blob_name);
        if (layers_[i]->layer_param().has_quantization_param()) {
          const QuantizationParameter& qparam = layers_[i]->layer_param().quantization_param();
          if (qparam.quantize_layer_out()) {
            blob_name += " : quantized";
            layer_blob_log->set_is_quantized(true);
            layer_blob_log->set_is_unsigned(qparam.unsigned_layer_out());
            layer_blob_log->set_bw(qparam.bw_layer_out());
            layer_blob_log->set_fl(qparam.fl_layer_out());
            Convert2FixedPoint_cpu(blob_copy.mutable_cpu_data(), blob_copy.count(), qparam.bw_layer_out(),
                qparam.fl_layer_out(), qparam.unsigned_layer_out(), true);
          }
        } else if(layers_[i]->layer_param().type() == "ArgMax") {
          blob_name += " : quantized";
          layer_blob_log->set_is_quantized(true);
          layer_blob_log->set_is_unsigned(true);
          layer_blob_log->set_bw(8);
          layer_blob_log->set_fl(0);
          Convert2FixedPoint_cpu(blob_copy.mutable_cpu_data(), blob_copy.count(), 8,
              0, true, true);
        }

        blob_copy.ToProto(blob_data, false);
      }
    }
  }
}

template<typename Dtype>
void Net<Dtype>::Convert2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width, int fl, bool unsigned_data, bool clip) const {
  for (int index = 0; index < cnt; ++index) {
    data[index] = data[index] * powf(2, fl);
    // Saturate data
#if CLIP_QUANT
      if(clip) {
          int qrange = unsigned_data? bit_width :  (bit_width - 1);
          Dtype max_data = +(powf(2, qrange) - 1);
          Dtype min_data = unsigned_data? 0 : -(powf(2, qrange));
          data[index] = std::max(std::min(data[index], max_data), min_data);
      }
#endif
    data[index] = round(data[index]);
    //data[index] = data[index] * pow(2, -fl);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}


template <typename Dtype>
void Net<Dtype>::ClearQuantizationRangeInLayers() {
  max_in_.clear();
  max_out_.clear();
  max_weights_.clear();

  min_in_.clear();
  min_out_.clear();
  min_weights_.clear();
}

template <typename Dtype>
void Net<Dtype>::CopyQuantizationRangeInLayers() {
  max_in_.resize(layers_.size());
  max_out_.resize(layers_.size(), 0);
  max_weights_.resize(layers_.size(), 0);

  min_in_.resize(layers_.size());
  min_out_.resize(layers_.size(), 0);
  min_weights_.resize(layers_.size(), 0);

  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    min_in_[layer_id].resize(bottom_vecs_[layer_id].size(), 0);
    max_in_[layer_id].resize(bottom_vecs_[layer_id].size(), 0);
  }

  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    if(!layers_[layer_id]->layer_param().has_quantization_param()) {
      continue;
    }
    const QuantizationParameter& source_quantization_param = layers_[layer_id]->layer_param().quantization_param();
    for(int blob_id = 0; blob_id<min_in_[layer_id].size(); blob_id++) {
      if(source_quantization_param.min_layer_in_size() > blob_id) {
        min_in_[layer_id][blob_id] = source_quantization_param.min_layer_in(blob_id);
      }
    }
    for(int blob_id = 0; blob_id<max_in_[layer_id].size(); blob_id++) {
      if(source_quantization_param.max_layer_in_size() > blob_id) {
        max_in_[layer_id][blob_id] = source_quantization_param.max_layer_in(blob_id);
      }
    }

    min_out_[layer_id] = source_quantization_param.min_layer_out();
    max_out_[layer_id] = source_quantization_param.max_layer_out();

    min_weights_[layer_id] = source_quantization_param.min_layer_weights();
    max_weights_[layer_id] = source_quantization_param.max_layer_weights();
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateQuantizationRangeInLayers() {
  max_in_.resize(layers_.size());
  max_out_.resize(layers_.size(), 0);
  max_weights_.resize(layers_.size(), 0);

  min_in_.resize(layers_.size());
  min_out_.resize(layers_.size(), 0);
  min_weights_.resize(layers_.size(), 0);

  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    min_in_[layer_id].resize(bottom_vecs_[layer_id].size(), 0);
    max_in_[layer_id].resize(bottom_vecs_[layer_id].size(), 0);
  }

  // Find maximal values.
  Dtype alpha = 0.99;
  Dtype beta = (1.0 - alpha);
  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
	if(bottom_vecs_[layer_id].size()>0) {
		for(int blob_id = 0; blob_id<bottom_vecs_[layer_id].size(); blob_id++) {
		    Dtype min_in = bottom_vecs_[layer_id][blob_id]->min();
		    Dtype max_in = bottom_vecs_[layer_id][blob_id]->max();
            min_in_[layer_id][blob_id] = min_in_[layer_id][blob_id] * alpha +  min_in * beta;
            max_in_[layer_id][blob_id] = max_in_[layer_id][blob_id] * alpha +  max_in * beta;
		}
	}

    Dtype min_out = std::numeric_limits<Dtype>::max();
	Dtype max_out = std::numeric_limits<Dtype>::min();
	if(top_vecs_[layer_id].size() > 0) {
		for(int blob_id = 0; blob_id<top_vecs_[layer_id].size(); blob_id++) {
          min_out = std::min(min_out, top_vecs_[layer_id][blob_id]->min());
		  max_out = std::max(min_out, top_vecs_[layer_id][blob_id]->max());
		}
        min_out_[layer_id] = min_out_[layer_id] * alpha + min_out * beta;
		max_out_[layer_id] = max_out_[layer_id] * alpha + max_out * beta;
	}

	//TODO: Set to 1 to consider the weights only, and ignore the bias
	int max_params_to_consider = 1;//INT_MAX;
	int num_params = std::min((int)layers_[layer_id]->blobs().size(), max_params_to_consider);
    Dtype min_weights = std::numeric_limits<Dtype>::max();
	Dtype max_weights = std::numeric_limits<Dtype>::min();
	if(num_params > 0) {
		for(int blob_id = 0; blob_id < num_params; blob_id++) {
          min_weights = std::min(min_weights, (Dtype)layers_[layer_id]->blobs()[blob_id]->min());
		  max_weights = std::max(max_weights, (Dtype)layers_[layer_id]->blobs()[blob_id]->max());
		}
        min_weights_[layer_id] = min_weights_[layer_id] * alpha + min_weights * beta;
		max_weights_[layer_id] = max_weights_[layer_id] * alpha + max_weights * beta;
	}
  }
}

template<typename Dtype>
void Net<Dtype>::SetTrainQuantizationParamsLayerInput(const int layer_id, QuantizationParameter_Precision precision,
    QuantizationParameter_Rounding rounding_scheme, const int bw_conv, const int bw_fc, const int bw_in,
    const int bw_out, bool unsigned_check_in, bool unsigned_check_out, Dtype sparsity_threshold) {
  QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
  if(true) { // (quantization_param.quantize_layer_in()) {
    int num_bottom_vecs = bottom_vecs_[layer_id].size();
    for(int blob_id = 0; blob_id<num_bottom_vecs; blob_id++) {
      bool unsigned_layer_in;
      Dtype min_layer_in, max_layer_in;
      int len_in = GetIntegerLengthIn(layer_id, blob_id, unsigned_check_in, unsigned_layer_in,
          min_layer_in, max_layer_in);
      quantization_param.set_precision(precision);
      quantization_param.set_rounding_scheme(rounding_scheme);
      if(quantization_param.fl_layer_in_size() > blob_id) {
        quantization_param.set_fl_layer_in(blob_id, bw_in - len_in);
      } else {
        quantization_param.add_fl_layer_in(bw_in - len_in);
      }
      quantization_param.set_bw_layer_in(bw_in);

      if (unsigned_check_in) {
        if(quantization_param.unsigned_layer_in_size() < num_bottom_vecs) {
          quantization_param.add_unsigned_layer_in(unsigned_layer_in);
        } else {
          quantization_param.set_unsigned_layer_in(blob_id, unsigned_layer_in);
        }
      }

      if(quantization_param.min_layer_in_size() < num_bottom_vecs) {
        quantization_param.add_min_layer_in(min_layer_in);
      } else {
        quantization_param.set_min_layer_in(blob_id, min_layer_in);
      }

      if(quantization_param.max_layer_in_size() < num_bottom_vecs) {
        quantization_param.add_max_layer_in(max_layer_in);
      } else {
        quantization_param.set_max_layer_in(blob_id, max_layer_in);
      }
    }
  }
}

template<typename Dtype>
void Net<Dtype>::SetTrainQuantizationParamsLayerOutput(const int layer_id, QuantizationParameter_Precision precision,
    QuantizationParameter_Rounding rounding_scheme, const int bw_conv, const int bw_fc, const int bw_in,
    const int bw_out, bool unsigned_check_in, bool unsigned_check_out, Dtype sparsity_threshold) {
  QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
  if(true) { //(quantization_param.quantize_layer_out()) {
    bool unsigned_layer_out;
    Dtype min_layer_out, max_layer_out;
    int len_out = GetIntegerLengthOut(layer_id, unsigned_check_out, unsigned_layer_out,
        min_layer_out, max_layer_out);
    quantization_param.set_fl_layer_out(bw_out - len_out);
    quantization_param.set_bw_layer_out(bw_out);
    if (unsigned_check_out) {
      quantization_param.set_unsigned_layer_out(unsigned_layer_out);
    }
    quantization_param.set_min_layer_out(min_layer_out);
    quantization_param.set_max_layer_out(max_layer_out);
  }

  if(quantization_param.quantize_layer_out()) {
    int fl_in = 0, fl_weights = 0, fl_out = 0;
    if(layers_[layer_id]->layer_param().type() == "Convolution" ||
        layers_[layer_id]->layer_param().type() == "Deconvolution") {
      fl_in = quantization_param.fl_layer_in(0);
      fl_weights = quantization_param.fl_weights();
      fl_out = quantization_param.fl_layer_out();
    } else if(layer_id>0 && layers_[layer_id]->layer_param().type() == "ReLU" &&
      layers_[layer_id-1]->layer_param().type() == "Convolution") {
      if(layers_[layer_id-1]->mutable_layer_param().has_quantization_param()) {
        QuantizationParameter& quantization_param_prev = *layers_[layer_id-1]->mutable_layer_param().mutable_quantization_param();
        fl_in = quantization_param_prev.fl_layer_in(0);
        fl_weights = quantization_param_prev.fl_weights();
        fl_out = quantization_param.fl_layer_out();
      } else {
        LOG(FATAL) << "Couldn't verify quantization params for layers: "
            << layers_[layer_id-1]->layer_param().name()
            << " and " << layers_[layer_id]->layer_param().name();
      }
    }
    if((fl_in + fl_weights) < fl_out) {
      LOG(FATAL) << "Qformat error for layer: "
          << layers_[layer_id]->layer_param().name() << "  fl_in:" << fl_in << " fl_weights:" << fl_weights
          << " fl_out:" << fl_out;
    }
  }
}

template<typename Dtype>
void Net<Dtype>::SetTrainQuantizationParamsLayerParams(const int layer_id, QuantizationParameter_Precision precision,
    QuantizationParameter_Rounding rounding_scheme, const int bw_conv, const int bw_fc, const int bw_in,
    const int bw_out, bool unsigned_check_in, bool unsigned_check_out, Dtype sparsity_threshold) {
  QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
  if(layers_[layer_id]->blobs().size() > 0) { // (quantization_param.quantize_layer_weights()) {
    Dtype min_layer_weights, max_layer_weights;
    int len_params = GetIntegerLengthWeights(layer_id, min_layer_weights, max_layer_weights);
    quantization_param.set_precision(precision);
    quantization_param.set_rounding_scheme(rounding_scheme);
    quantization_param.set_fl_weights(bw_conv - len_params);
    quantization_param.set_bw_weights(bw_conv);
    quantization_param.set_min_layer_weights(min_layer_weights);
    quantization_param.set_max_layer_weights(max_layer_weights);
    // Sparsity is handled in Net after the Update step
    //quantization_param->set_sparsity_threshold(sparsity_threshold);
  }
}

template<typename Dtype>
void Net<Dtype>::SetTrainQuantizationParams(QuantizationParameter_Precision precision,
    QuantizationParameter_Rounding rounding_scheme, const int bw_conv, const int bw_fc, const int bw_in,
    const int bw_out, bool unsigned_check_in, bool unsigned_check_out, Dtype sparsity_threshold,
    bool quantize_weights, bool quantize_activations) {

  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    if (layers_[layer_id]->layer_param().has_quantization_param()) {
      // quantize parameters
      if(true) { //if (quantize_weights) {
        SetTrainQuantizationParamsLayerParams(layer_id, precision, rounding_scheme, bw_conv, bw_fc, bw_in, bw_out, unsigned_check_in,
            unsigned_check_out, sparsity_threshold);
      }

      // quantize input activations
      if(true) { //if (quantize_activations) {
        SetTrainQuantizationParamsLayerInput(layer_id, precision, rounding_scheme, bw_conv, bw_fc, bw_in, bw_out, unsigned_check_in,
            unsigned_check_out, sparsity_threshold);
      }

      // quantize output activations
      if(true) { //if (quantize_activations) {
        SetTrainQuantizationParamsLayerOutput(layer_id, precision, rounding_scheme, bw_conv, bw_fc, bw_in, bw_out, unsigned_check_in,
            unsigned_check_out, sparsity_threshold);
      }
    }
  }
}

template<typename Dtype>
void Net<Dtype>::SetTestQuantizationParams(QuantizationParameter_Precision precision,
    QuantizationParameter_Rounding rounding_scheme, const int bw_conv, const int bw_fc, const int bw_in,
    const int bw_out, bool unsigned_check_in, bool unsigned_check_out, Dtype sparsity_threshold, bool quantize_weights,
    bool quantize_activations) {
  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    if (layers_[layer_id]->layer_param().has_quantization_param()) {
      QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
      quantization_param.set_rounding_scheme(rounding_scheme);
    }
  }
}

template<typename Dtype>
void Net<Dtype>::DisplayQuantizationParams(bool quantize_weights, bool quantize_activations) {
  for (int i = 0; i < layers_.size(); ++i) {
    if (layers_[i]->layer_param().has_quantization_param()) {
      // if this is a convolutional layer which should be quantized ...
      QuantizationParameter& quantization_param = *layers_[i]->mutable_layer_param().mutable_quantization_param();
      if (quantize_weights && quantization_param.quantize_layer_weights()) {
        LOG(INFO)<<" Q params:" << i << " Name:" << layers_[i]->layer_param().name() <<
        " bw_weights:" << quantization_param.bw_weights() <<
        " fl_weights:" << quantization_param.fl_weights() <<
        " min_weights:" << quantization_param.min_layer_weights() <<
        " max_weights:" << quantization_param.max_layer_weights();
      }

      if (quantize_activations && quantization_param.quantize_layer_in()) {
        int num_bottom_vecs = bottom_vecs_[i].size();
        std::stringstream ss;
        ss << " Q input :" << i << " Name:" << layers_[i]->layer_param().name() <<
        " bw_in:" << quantization_param.bw_layer_in();
        for(int blob_id=0; blob_id<std::min<int>(num_bottom_vecs, quantization_param.fl_layer_in_size()); blob_id++) {
          ss << " fl_in:" << quantization_param.fl_layer_in(blob_id);
        }
        for(int blob_id=0; blob_id<std::min<int>(num_bottom_vecs, quantization_param.min_layer_in_size()); blob_id++) {
          ss << " min_in:" << quantization_param.min_layer_in(blob_id);
        }
        for(int blob_id=0; blob_id<std::min<int>(num_bottom_vecs, quantization_param.max_layer_in_size()); blob_id++) {
          ss << " max_in:" << quantization_param.max_layer_in(blob_id);
        }
        for(int blob_id=0; blob_id<std::min<int>(num_bottom_vecs, quantization_param.unsigned_layer_in_size()); blob_id++) {
          ss << " unsigned_in:" << quantization_param.unsigned_layer_in(blob_id);
        }
        LOG(INFO) << ss.str();
      }

      if (quantize_activations && quantization_param.quantize_layer_out()) {
        LOG(INFO)<< " Q output:" << i << " Name:" << layers_[i]->layer_param().name() <<
        " bw_out:" << quantization_param.bw_layer_out() <<
        " fl_out:" << quantization_param.fl_layer_out() <<
        " min_out:" << quantization_param.min_layer_out() <<
        " max_out:" << quantization_param.max_layer_out() <<
        " unsigned_out:" << quantization_param.unsigned_layer_out();
      }
    }
  }
}

template<typename Dtype>
void Net<Dtype>::DisableQuantization(bool quantize_weights, bool quantize_activations) {
  for (int i = 0; i < layers_.size(); ++i) {
    if (layers_[i]->layer_param().has_quantization_param()) {
      QuantizationParameter& quantization_param = *layers_[i]->mutable_layer_param().mutable_quantization_param();
      quantization_param.set_precision(QuantizationParameter_Precision_FLOAT);
    }
  }
}
	  
template<typename Dtype>
void Net<Dtype>::AddQuantizationParams() {
  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    if(layers_[layer_id]->layer_param().type() == "Convolution" || layers_[layer_id]->layer_param().type() == "InnerProduct") {
      QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
      quantization_param.set_quantize_layer_weights(true);
      if((layer_id+1) < layers_.size() && layers_[layer_id+1]->layer_param().type() != "ReLU") {
        quantization_param.set_quantize_layer_out(true);
      }
    } else if(layers_[layer_id]->layer_param().type() == "ReLU" || layers_[layer_id]->layer_param().type() == "Scale" ||
        layers_[layer_id]->layer_param().type() == "Pooling") {
      QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
      quantization_param.set_quantize_layer_out(true);
    } else if(layers_[layer_id]->layer_param().type() == "Eltwise") {
      if((layer_id+1) < layers_.size() && layers_[layer_id+1]->layer_param().type() != "ReLU") {
        QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
        quantization_param.set_quantize_layer_out(true);
      }
    } else if(layers_[layer_id]->layer_param().type() == "Concat") {
      QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
      quantization_param.set_quantize_layer_out(true);
    } if(layers_[layer_id]->layer_param().type() == "Deconvolution") {
      QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
      quantization_param.set_quantize_layer_weights(true);
      quantization_param.set_quantize_layer_out(true);
    }
  }
}

template <typename Dtype>
int Net<Dtype>::EstimateAbsBits(Dtype val) {
	return ceil(log2(std::fabs(val)));
}

template <typename Dtype>
int Net<Dtype>::GetIntegerLengthWeights(const int layer_id, Dtype& min_layer_weights, Dtype& max_layer_weights) {
  min_layer_weights = min_weights_[layer_id];
  max_layer_weights = max_weights_[layer_id];
  return EstimateAbsBits(max_weights_[layer_id]) + 1;
}

template <typename Dtype>
int Net<Dtype>::GetIntegerLengthIn(const int layer_id, const int blob_id, bool unsigned_check_in,
		bool& unsigned_layer_in, Dtype& min_layer_in, Dtype& max_layer_in) {
  min_layer_in = min_in_[layer_id][blob_id];
  max_layer_in = max_in_[layer_id][blob_id];
  Dtype max_val_abs = std::max(std::fabs(max_layer_in), std::fabs(min_layer_in));
  Dtype min_val = min_layer_in;
  unsigned_layer_in = unsigned_check_in? min_val>=0 : false;
  return (unsigned_check_in && unsigned_layer_in)?
		  EstimateAbsBits(max_val_abs) : (EstimateAbsBits(max_val_abs) + 1);
}

template <typename Dtype>
int Net<Dtype>::GetIntegerLengthOut(const int layer_id, bool unsigned_check_out,
		bool& unsigned_layer_out, Dtype& min_layer_out, Dtype& max_layer_out) {
  min_layer_out = min_out_[layer_id];
  max_layer_out = max_out_[layer_id];
  Dtype max_val_abs = std::max(std::fabs(max_layer_out), std::fabs(min_layer_out));
  Dtype min_val = min_layer_out;
  unsigned_layer_out = unsigned_check_out? min_val>=0 : false;
  return (unsigned_check_out && unsigned_layer_out)?
		  EstimateAbsBits(max_val_abs) : (EstimateAbsBits(max_val_abs) + 1);
}

template <typename Dtype>
void Net<Dtype>::OptimizeNet() {
  for (int i = 0; i < layers_.size(); i++) {
    if(layers_[i]->layer_param().has_quantization_param() &&
        layers_[i]->layer_param().quantization_param().has_rounding_scheme()) {
      QuantizationParameter& qparam = *layers_[i]->mutable_layer_param().mutable_quantization_param();
      qparam.set_rounding_scheme(QuantizationParameter_Rounding_NEAREST);
    }
  }

  bool can_merge_bn = false;
  for (int i = 0; i < (layers_.size()-1); i++) {
    if (layers_[i]->type() == std::string("Convolution") &&
        layers_[i+1]->type() == std::string("BatchNorm")) {
      can_merge_bn = true;
    }
  }

  if(!can_merge_bn) {
    return;
  }

  for (int i = 0; i < (layers_.size()-1); i++) {
    if (layers_[i]->type() == std::string("Convolution")) {
      Layer<Dtype>& conv_layer = *layers_[i];
      Blob<Dtype>& conv_weights = *conv_layer.blobs()[0];
      int channels = (conv_weights.num_axes() == 1)? conv_weights.count() : conv_weights.shape(0);
      int outputs = channels;

      // Set bias term if it not there, as it is needed when conbining BN
      if(conv_layer.blobs().size()==1) {
        bool bias_term = true;
        conv_layer.mutable_layer_param().mutable_convolution_param()->set_bias_term(bias_term);
        conv_layer.mutable_layer_param().mutable_convolution_param()->mutable_bias_filler()->set_type("constant");
        conv_layer.mutable_layer_param().mutable_convolution_param()->mutable_bias_filler()->set_value(0);

        conv_layer.blobs().resize(2);
        vector<int> bias_shape(bias_term, outputs);
        conv_layer.blobs()[1].reset(new Blob<Dtype>(bias_shape));
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
            conv_layer.layer_param().convolution_param().bias_filler()));
        bias_filler->Fill(conv_layer.blobs()[1].get());
      }

      if(layers_[i+1]->type() == std::string("BatchNorm")) {
        Layer<Dtype>& batch_norm_layer = *layers_[i+1];

        Blob<Dtype>& batch_norm_scale = *batch_norm_layer.blobs()[0];
        Blob<Dtype>& batch_norm_bias = *batch_norm_layer.blobs()[1];
        Blob<Dtype>& batch_norm_mean = *batch_norm_layer.blobs()[2];
        Blob<Dtype>& batch_norm_var = *batch_norm_layer.blobs()[3];
        Dtype eps = batch_norm_layer.layer_param().batch_norm_param().eps();

        // Absorb the BatchNorm into convolution
        for(int no=0; no<conv_weights.shape(0); no++) {
          Dtype var = batch_norm_var.data_at(no) + eps;
          Dtype stdev_inv = std::pow(var, Dtype(-0.5));
          Dtype scale = batch_norm_scale.data_at(no);
          for(int ni=0; ni<conv_weights.shape(1); ni++) {
            for(int w=0; w<conv_weights.shape(2); w++) {
              for(int h=0; h<conv_weights.shape(3); h++) {
                conv_weights.data_at(no,ni,w,h) = conv_weights.data_at(no,ni,w,h) * stdev_inv * scale;
              }
            }
          }
        }

        Blob<Dtype>& conv_bias = *conv_layer.blobs()[1];
        for(int no=0; no<channels; no++) {
          Dtype var = batch_norm_var.data_at(no) + eps;
          Dtype stdev_inv = std::pow(var, Dtype(-0.5));
          Dtype scale = batch_norm_scale.data_at(no);
          Dtype bias = batch_norm_bias.data_at(no);
          Dtype mean = batch_norm_mean.data_at(no);
          conv_bias.data_at(no) = (conv_bias.data_at(no) - mean) * stdev_inv * scale + bias;
        }

        // Set the batch norm to identity
        for(int c=0; c<channels; c++) {
          batch_norm_scale.data_at(c) = Dtype(1.0);
          batch_norm_bias.data_at(c) = Dtype(0.0);
          batch_norm_mean.data_at(c) = Dtype(0.0);
          //Change var so that after adding eps, it becomes 1.0
          batch_norm_var.data_at(c) = Dtype(1.0 - eps);
        }
      }
    }
  }

  //Merge a BatchNorm layer that comes before convolution layer
  for (int i = 0; i < (layers_.size()-1); i++) {
    if (layers_[i]->type() == std::string("BatchNorm") && layers_[i+1]->type() == std::string("Convolution")) {
      Layer<Dtype>& batch_norm_layer = *layers_[i];
      Layer<Dtype>& conv_layer = *layers_[i+1];
      Blob<Dtype>& conv_weights = *conv_layer.blobs()[0];
      Blob<Dtype>& conv_bias = *conv_layer.blobs()[1];
      int channels = (conv_weights.num_axes() == 1)? conv_weights.count() : conv_weights.shape(0);

      Blob<Dtype>& batch_norm_scale = *batch_norm_layer.blobs()[0];
      Blob<Dtype>& batch_norm_bias = *batch_norm_layer.blobs()[1];
      Blob<Dtype>& batch_norm_mean = *batch_norm_layer.blobs()[2];
      Blob<Dtype>& batch_norm_var = *batch_norm_layer.blobs()[3];
      Dtype eps = batch_norm_layer.layer_param().batch_norm_param().eps();

      // Absorb the BatchNorm into convolution
      for(int no=0; no<conv_weights.shape(0); no++) {
        Dtype var = batch_norm_var.data_at(no) + eps;
        Dtype stdev_inv = std::pow(var, Dtype(-0.5));
        Dtype scale = batch_norm_scale.data_at(no);
        Dtype bias = batch_norm_bias.data_at(no);
        Dtype mean = batch_norm_mean.data_at(no);

        Dtype weight_sum = 0;
        for(int ni=0; ni<conv_weights.shape(1); ni++) {
          for(int w=0; w<conv_weights.shape(2); w++) {
            for(int h=0; h<conv_weights.shape(3); h++) {
              weight_sum += conv_weights.data_at(no,ni,w,h);
              conv_weights.data_at(no,ni,w,h) = conv_weights.data_at(no,ni,w,h) * stdev_inv * scale;
            }
          }
        }
        conv_bias.data_at(no) = conv_bias.data_at(no) + bias * weight_sum - mean * stdev_inv * weight_sum;
      }

      // Set the batch norm to identity
      for(int c=0; c<channels; c++) {
        batch_norm_scale.data_at(c) = Dtype(1.0);
        batch_norm_bias.data_at(c) = Dtype(0.0);
        batch_norm_mean.data_at(c) = Dtype(0.0);
        //Change var so that after adding eps, it becomes 1.0
        batch_norm_var.data_at(c) = Dtype(1.0 - eps);
      }
    }
  }

}


template <typename Dtype>
void Net<Dtype>::ThresholdNet(float threshold_fraction_low, float threshold_fraction_mid, float threshold_fraction_high,
    float threshold_value_maxratio, float threshold_value_max, float threshold_step_factor) {
  for (int i = 0; i < layers_.size(); i++) {
    if(layers_[i]->layer_param().has_quantization_param() &&
        layers_[i]->layer_param().quantization_param().has_rounding_scheme()) {
      QuantizationParameter& qparam = *layers_[i]->mutable_layer_param().mutable_quantization_param();
      qparam.set_rounding_scheme(QuantizationParameter_Rounding_NEAREST);
    }
  }

  for (int i = 0; i < (layers_.size()-1); i++) {
    if (layers_[i]->type() == std::string("Convolution")) {
      Layer<Dtype>& conv_layer = *layers_[i];
      Blob<Dtype>& conv_weights = *conv_layer.blobs()[0];
      int num_group = layers_[i]->layer_param().convolution_param().group();
      //int stride = layers_[i]->layer_param().convolution_param().stride_size()>0? layers_[i]->layer_param().convolution_param().stride(0) : 1;

      int no = (conv_weights.num_axes() == 1)? conv_weights.count() : conv_weights.shape(0);
      int ni = ((conv_weights.num_axes() == 1)? conv_weights.count() : conv_weights.shape(1))*num_group;
      float count = conv_weights.count();
      LOG(WARNING) << layers_[i]->layer_param().name() << " ni=" << ni << " no=" << no;

      if(ni>=32 || no >= 32) {
        float threshold_fraction_selected = ((ni>=256 && no >= 512)? threshold_fraction_high :
            ((ni>=32 && no >= 32)? threshold_fraction_mid: threshold_fraction_low));
        float selected_threshold = 0;
        float max_abs = std::abs(conv_weights.max());
        float min_abs = std::abs(conv_weights.min());
        float max_abs_value = std::max<float>(max_abs, min_abs);
        float step_size = max_abs_value * threshold_step_factor;
        float max_threshold_value = std::min<float>(threshold_value_max, max_abs_value*threshold_value_maxratio);

        //LOG(WARNING) << layers_[i]->layer_param().name() << " MaxAbsWeight=" << max_abs_value;
        //LOG(WARNING) << layers_[i]->layer_param().name() << " max_threshold_value=" << max_threshold_value;
        //LOG(WARNING) << layers_[i]->layer_param().name() << " step_size=" << step_size;

        for(float step=0; step<max_abs_value && step<max_threshold_value; step+=step_size) {
          float zcount = conv_weights.count_zero((Dtype)step);
          float zratio = zcount / count;
          //LOG(WARNING) << layers_[i]->layer_param().name() << " Threshold=" << step << " ZeroPercentage=" << zratio*100;
          if(zratio <= threshold_fraction_selected) {
            selected_threshold = step;
          } else {
            break;
          }
        }

        conv_weights.Zerout(selected_threshold);
        float zcount = conv_weights.count_zero(selected_threshold);
        LOG(WARNING) << layers_[i]->layer_param().name() << " MaxAbsWeight=" << max_abs_value << " MaxThreshold=" << max_threshold_value << " SelectedThreshold=" << selected_threshold << " ZeroPercentage=" << (zcount*100/count);
      }
    }
  }

  this->DisplaySparsity(0.0);
}

template <typename Dtype>
void Net<Dtype>::DisplaySparsity(float sparsity_threshold) {
std::map<std::string, std::pair<int,int> > spasity_map;
  int blob_count = this->GetSparsity(spasity_map, sparsity_threshold);
  LOG(INFO) << "Num Params(" << blob_count << "), ";
  Dtype total_zero_count = 0, total_count = 0;
  std::stringstream ss;
  ss << "Convolution and InnerProduct Layers, Sparsity (zero_weights/count): ";
  for(std::map<std::string, std::pair<int,int> >::iterator
      iter = spasity_map.begin(); iter != spasity_map.end(); iter++) {
    std::string param_name = iter->first;
    Dtype zero_count = iter->second.first;
    Dtype count = iter->second.second;
    total_zero_count += zero_count;
    total_count += count;
    ss << param_name << "(" << std::setprecision(3) << (zero_count/count) << ") ";
    //ss << param_name << "(" << zero_count << "/" << count << ") ";
  }
  LOG(INFO) << ss.str();
  LOG(INFO) << "Total Sparsity (zero_weights/count) = "
      << " (" << total_zero_count << "/" << total_count << ") "
      << std::setprecision(3) << (total_zero_count/total_count);
}

template <typename Dtype>
void Net<Dtype>::SetSparseMode(SparseMode mode) {
  for(int layer_id=0; layer_id<layers_.size(); layer_id++) {
    layers_[layer_id]->SetSparseMode(mode);
  }
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
