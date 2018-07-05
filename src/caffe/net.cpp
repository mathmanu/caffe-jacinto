#include <algorithm>
#include <map>
#include <set>
#include <boost/thread.hpp>
#include <hdf5.h>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

constexpr int Net::END_OF_ITERATION;
constexpr int Net::END_OF_TRAIN;

Net::Net(const NetParameter& param,
    size_t solver_rank,
    Flag* solver_init_flag,
    const Net* root_net,
    bool inner_net,
    int level,
    const vector<string>* stages)
    : root_net_(root_net),
      solver_(nullptr),
      solver_rank_(solver_rank),
      solver_init_flag_(solver_init_flag),
      inner_net_(inner_net) {
  Init(param);
}

Net::Net(const string& param_file,
    Phase phase,
    size_t solver_rank,
    Flag* solver_init_flag,
    const Net* root_net,
    bool inner_net,
    int level,
    const vector<string>* stages)
    : root_net_(root_net),
      solver_(nullptr),
      solver_rank_(solver_rank),
      solver_init_flag_(solver_init_flag),
      inner_net_(inner_net) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // Set phase, stages and level
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); ++i) {
      param.mutable_state()->add_stage(stages->at(i));
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
}

Net::~Net() {
}

void Net::Init(const NetParameter& in_param) {
  CHECK(inner_net_ || Caffe::root_solver() || root_net_)
      << "root_net_ needs to be set for all non-root solvers";
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  net_param_ = filtered_param;
  batch_per_solver_ = caffe::P2PSync::divide_batch_size(&filtered_param);
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  infer_count_ = 0UL;
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  gpu_top_memory_data_use_ = gpu_top_memory_diff_use_ = 0UL;
  gpu_btm_memory_data_use_ = gpu_btm_memory_diff_use_ = 0UL;
  gpu_shr_memory_data_use_ = gpu_shr_memory_diff_use_ = 0UL;
  gpu_prm_memory_data_use_ = gpu_prm_memory_diff_use_ = 0UL;
  gpu_shp_memory_data_use_ = gpu_shp_memory_diff_use_ = 0UL;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());

  // If user skips default math type we use default data type:
  Type default_fmath, default_bmath;
  if (in_param.has_default_forward_math()) {
    default_fmath = in_param.default_forward_math();
  } else {
    default_fmath = in_param.default_forward_type();
    LOG(INFO) << "Using " << Type_Name(default_fmath) << " as default forward math type";
  }
  if (in_param.has_default_backward_math()) {
    default_bmath = in_param.default_backward_math();
  } else {
    default_bmath = in_param.default_backward_type();
    LOG(INFO) << "Using " << Type_Name(default_bmath) << " as default backward math type";
  }

  wgrad_sq_.store(0LL);
  global_grad_scale_coeff_ = 1.F;
  has_global_grad_scale_param_ = in_param.has_global_grad_scale();
  global_grad_scale_param_ = in_param.global_grad_scale();
  global_grad_scale_adaptive_ = in_param.global_grad_scale_adaptive();

  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // For non-root solvers, whether this layer is shared from root_net_.
    bool share_from_root = !inner_net_ && !Caffe::root_solver()
        && root_net_->layers_[layer_id]->ShareInParallel();

    const LayerParameter& layer_param = param.layer(layer_id);
    LayerParameter* mutable_layer_param = param.mutable_layer(layer_id);

    DLOG_IF(INFO, Caffe::root_solver())
        << "Setting types for Layer " << layer_param.name();

    // Inherit phase from net if unset.
    if (!layer_param.has_phase()) {
      mutable_layer_param->set_phase(phase_);
    }
    const bool is_data_layer = layer_param.has_transform_param();

    // Data&Math types
    const bool fm_by_user = layer_param.has_forward_math();
    if (!fm_by_user) {
      if (layer_param.has_forward_type()) {
        mutable_layer_param->set_forward_math(layer_param.forward_type());
      } else {
        mutable_layer_param->set_forward_math(default_fmath);
      }
    }
    const bool bm_by_user = layer_param.has_backward_math();
    if (!bm_by_user) {
      if (layer_param.has_backward_type()) {
        mutable_layer_param->set_backward_math(layer_param.backward_type());
      } else {
        mutable_layer_param->set_backward_math(default_bmath);
      }
    }

    if (!layer_param.has_forward_type()) {
      mutable_layer_param->set_forward_type(in_param.default_forward_type());
    }
    if (!layer_param.has_backward_type()) {
      if (is_data_layer) {
        // In majority of cases we manage to avoid redundant conversion:
        mutable_layer_param->set_backward_type(FLOAT);
      } else {
        mutable_layer_param->set_backward_type(in_param.default_backward_type());
      }
    }

    // Convolution algorithms
    if (param.has_default_conv_algos_override() && layer_param.has_convolution_param() &&
        !layer_param.convolution_param().has_conv_algos_override()) {
      mutable_layer_param->mutable_convolution_param()->
          set_conv_algos_override(param.default_conv_algos_override());
    }

    // cuDNN math
    if (param.has_default_cudnn_math_override() &&
        !layer_param.has_cudnn_math_override()) {
      mutable_layer_param->set_cudnn_math_override(param.default_cudnn_math_override());
    }

    // Setup layer.
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
      layers_.push_back(LayerRegistry::CreateLayer(layer_param, solver_rank_));
    }
    layer_names_.push_back(layer_param.name());
    LOG_IF(INFO, Caffe::root_solver())
        << "Created Layer " << layer_param.name() << " (" << layer_id << ")";
    bool need_backward = false;

    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id) {
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
    LayerBase* layer = layers_[layer_id].get();
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
    layer->fm_by_user(fm_by_user);
    layer->bm_by_user(bm_by_user);

    layers_[layer_id]->set_net_initialized_flag(solver_init_flag_);

    Flag* layer_inititialized_flag = layers_[layer_id]->layer_inititialized_flag();
    if (layer_inititialized_flag != nullptr) {
      layer_inititialized_flags_.push_back(layer_inititialized_flag);
    }

    // After this layer is connected, set it up.
    if (share_from_root) {
      // Set up size of top blobs using root_net_
      const vector<Blob*>& base_top = root_net_->top_vecs_[layer_id];
      const vector<Blob*>& this_top = this->top_vecs_[layer_id];
      for (int top_id = 0; top_id < base_top.size(); ++top_id) {
        this_top[top_id]->ReshapeLike(*base_top[top_id]);
        LOG(INFO) << "Created top blob " << top_id << " (shape: "
            << this_top[top_id]->shape_string() <<  ") for shared layer "
            << layer_param.name();
      }
    } else {
      layers_[layer_id]->set_parent_net(this);
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, 0.F);
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG_IF(INFO, Caffe::root_solver())
          << Phase_Name(phase_) << " Top shape for layer " << layer_id << " '"
          << layer_names_[layer_id] << "' " <<  top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id) != 0.F) {
        LOG_IF(INFO, Caffe::root_solver())
          << "    with loss weight " << layer->loss(top_id);
      }
      gpu_top_memory_data_use_ += top_vecs_[layer_id][top_id]->gpu_memory_data_use();
      gpu_top_memory_diff_use_ += top_vecs_[layer_id][top_id]->gpu_memory_diff_use();
    }
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
  // because the skip_propagate_down param) and so we can skip backward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) != 0.F ||
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
  for (int blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (int layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();

  // invert param_layer_indices_ to give map of
  // (level_id, local param_id) -> global param_id
  for (int i = 0; i < param_layer_indices_.size(); ++i) {
    layer_index_params_[param_layer_indices_[i]] = i;
  }

  learnable_space_size_[0] = 0UL;
  learnable_space_size_[1] = 0UL;
  reduce_buckets_ = (size_t) in_param.reduce_buckets();
  if (Caffe::device_count() > 0) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Top memory (" << Phase_Name(phase_) << ") required for data: "
        << gpu_top_memory_data_use_ << " diff: " << gpu_top_memory_diff_use_;
    LOG_IF(INFO, Caffe::root_solver())
        << "Bottom memory (" << Phase_Name(phase_) << ") required for data: "
        << gpu_btm_memory_data_use_ << " diff: " << gpu_btm_memory_diff_use_;
    LOG_IF(INFO, Caffe::root_solver())
        << "Shared (in-place) memory (" << Phase_Name(phase_) << ") by data: "
        << gpu_shr_memory_data_use_ << " diff: " << gpu_shr_memory_diff_use_;
    LOG_IF(INFO, Caffe::root_solver())
        << "Parameters memory (" << Phase_Name(phase_) << ") required for data: "
        << gpu_prm_memory_data_use_ << " diff: " << gpu_prm_memory_diff_use_;
    LOG_IF(INFO, Caffe::root_solver())
        << "Parameters shared memory (" << Phase_Name(phase_) << ") by data: "
        << gpu_shp_memory_data_use_ << " diff: " << gpu_shp_memory_diff_use_;
  }
  debug_info_ = param.debug_info();
  trained_layers_shared_ = false;
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

void Net::FilterNet(const NetParameter& param, NetParameter* param_filtered) {
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

bool Net::StateMeetsRule(const NetState& state,
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
void Net::AppendTop(const NetParameter& param, const int layer_id, const int top_id,
    set<string>* available_blobs, map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = (layer_param.top_size() > top_id) ?
      layer_param.top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param.bottom_size() > top_id &&
      blob_name == layer_param.bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param.name() << " -> " << blob_name << " (in-place)";
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
    gpu_shr_memory_data_use_ += top_vecs_[layer_id].back()->gpu_memory_data_use();
    gpu_shr_memory_diff_use_ += top_vecs_[layer_id].back()->gpu_memory_diff_use();
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param.name() << " -> " << blob_name;
    }

    Type ftype = layer_param.has_forward_type() ? layer_param.forward_type() :
        param.default_forward_type();
    Type btype = layer_param.has_backward_type() ? layer_param.backward_type() :
        param.default_backward_type();
    shared_ptr<Blob> blob_pointer = Blob::create(ftype, btype);
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
int Net::AppendBottom(const NetParameter& param, const int layer_id,
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
  gpu_btm_memory_data_use_ += bottom_vecs_[layer_id].back()->gpu_memory_data_use();
  gpu_btm_memory_diff_use_ += bottom_vecs_[layer_id].back()->gpu_memory_diff_use();
  return blob_id;
}

void Net::AppendParam(const NetParameter& param, const int layer_id, const int param_id) {
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
    learnable_params_.push_back(params_[net_param_id]);
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
    Blob* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob* owner_blob =
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

float Net::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());

  this->StartQuantization();

  float loss = 0;
  for (int i = start; i <= end; ++i) {
    // LOG(INFO) << " ****** [Forward] (" << i << ") Layer '" << layer_names_[i];
    // << "' FT " << Type_Name(layers_[i]->forward_type())
    // << " BT " << Type_Name(layers_[i]->backward_type());
    float layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }

  this->FinishQuantization();

  ++infer_count_;
  return loss;
}

float Net::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

float Net::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

const vector<Blob*>& Net::Forward(float* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

const vector<Blob*>& Net::Forward(const vector<Blob*>& bottom, float* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

float Net::ForwardBackward(bool apply_update) {
  float loss;
  Forward(&loss);
  Backward(apply_update);
  return loss;
}

void Net::BackwardFromTo(int start, int end) {
  BackwardFromToAu(start, end, true);
}

void Net::BackwardFromToAu(int start, int end, bool apply_update) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    if (!layer_need_backward_[i]) {
      continue;
    }

    layers_[i]->Backward(top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);

    if (debug_info_) {
      BackwardDebugInfo(i);
    }
    if (!apply_update) {
      continue;
    }
    for (int j = 0; j < layers_[i]->blobs().size(); ++j) {
      if (layers_[i]->skip_apply_update(j)) {
        continue;
      }
      const int param_id = layer_index_params_[make_pair(i, j)];
      if (param_owners_[param_id] < 0) {
        const int lparam_id = learnable_param_ids_[param_id];
        int t = (int)learnable_params_[lparam_id]->diff_type();
        for (int type_id = 0; type_id < learnable_types().size(); ++type_id) {
          if (t == learnable_types_[type_id]) {
            reduction_queue_[type_id].push(lparam_id);
            break;
          }
        }
      }  // leave it to the owner otherwise
    }
  }
  if (apply_update) {
    for (int type_id = 0; type_id < learnable_types_.size(); ++type_id) {
      reduction_queue_[type_id].push(END_OF_ITERATION);
    }
  }
}

void Net::Finalize() {
  for (int type_id = 0; type_id < learnable_types_.size(); ++type_id) {
    reduction_queue_[type_id].push(END_OF_TRAIN);
  }
}

size_t Net::received_contiguous_count(int type_id, const std::set<int>& au_ids, int& id_from_ret) {
  if (learnable_params_.empty() || au_ids.empty() || param_id_vecs_.empty()) {
    return 0;
  }
  size_t cnt_ret = 0UL, cnt = 0UL;
  const int bottom = *au_ids.begin();
  const int top = *au_ids.rbegin();
  id_from_ret = -1;
  const std::map<size_t, std::set<int>>& ltop = ltop_[type_id];
  for (auto lit = ltop.rbegin(); lit != ltop.rend(); ++lit) {
    if (lit->second.empty() || *lit->second.begin() > top) {
      continue;
    }
    bool layer_complete = true;
    for (auto p = lit->second.begin(); p != lit->second.end(); ++p) {
      int param_id = *p;
      if (param_id < bottom || au_ids.find(param_id) == au_ids.end()) {
        layer_complete = false;
        break;
      }
      cnt += lp_aligned_count(param_id);
    }
    if (layer_complete) {
      id_from_ret = *lit->second.begin();
      cnt_ret = cnt;
    } else {
      break;
    }
  }
  return cnt_ret;
}

void Net::ReduceAndUpdate(int type_id) {
  DLOG(INFO) << "[" << Caffe::current_device()
             << "] Entering ReduceAndUpdate thread " << lwp_id()
             <<  ", type_id " << type_id;

  size_t bucket_size = 0UL;
  cublasHandle_t handle = Caffe::cublas_handle(type_id);
  CHECK_GE(reduce_buckets_, 0);
  if (Caffe::solver_count() > 1 && reduce_buckets_ > 0) {
    bucket_size = align_up<6>(learnable_space_size_[type_id] / reduce_buckets_);
  }
  std::set<int> au_ids;

  const bool clip_grads = solver_->param().clip_gradients() >= 0.F;
  const bool clear_grads = !solver_->param().snapshot_diff() && !clip_grads;
  const bool use_buckets = reduce_buckets_ > 0;
  float rate = -1.F;
  while (!solver_->stop_reducing_requested(type_id)) {
    const int param_id = reduction_queue_[type_id].pop();
    SolverAction::Enum request = solver_->GetRequestedAction();
    if (SolverAction::STOP == request) {
      solver_->request_early_exit();
      break;
    }
    if (param_id == END_OF_TRAIN) {
      break;
    }
    if (rate < 0.F) {
      rate = solver_->GetLearningRate();
    }
    if (param_id != END_OF_ITERATION) {
      if (Caffe::solver_count() > 1) {
        if (!use_buckets && !clip_grads) {
          Reduce(type_id, param_id);
          if (solver_->stop_reducing_requested(type_id)) {
            break;
          }
          add_wgrad_sq(solver_->ApplyUpdate(param_id, handle, rate, true, clear_grads));
          continue;
        }
      } else {
        if (!clip_grads) {
          this->learnable_params()[param_id]->scale_diff(1.F / global_grad_scale(), handle);
          add_wgrad_sq(solver_->ApplyUpdate(param_id, handle, rate, true, clear_grads));
        }
        continue;
      }
    } else if (clip_grads && Caffe::solver_count() == 1) {
      solver_->ClipGradientsAndNormalize(handle, type_id, au_ids);
      for (int i : au_ids) {
        add_wgrad_sq(solver_->ApplyUpdate(i, handle, rate, false, clear_grads));
      }
      au_ids.clear();
    }

    if (!learnable_params_.empty() && Caffe::solver_count() > 1) {
      int id_from = -1;
      // Is bucket big enough? Done with iteration?
      const size_t received_count = received_contiguous_count(type_id, au_ids, id_from);
      if (id_from >= 0) {
        const size_t received_size = received_count * lp_size(id_from);
        if ((received_size >= bucket_size && !clip_grads) || param_id == END_OF_ITERATION) {
//#ifdef DEBUG
//          {
//            size_t c = 0UL;
//            for (int i : au_ids) {
//              if (i < id_from) {
//                continue;
//              }
//              c += lp_aligned_count(i);
//            }
//            CHECK_EQ(c, received_count);
//          }
//#endif
          CHECK_EQ((int) learnable_params_[id_from]->diff_type(), learnable_types_[type_id]);
          ReduceBucket(type_id, received_count, learnable_params_[id_from]->diff_type(),
              learnable_params_ptrs_[type_id][id_from]);
          if (solver_->stop_reducing_requested(type_id)) {
            break;
          }

          if (clip_grads) {
            solver_->ClipGradientsAndNormalize(handle, type_id, au_ids);
          }

          for (int i : au_ids) {
            add_wgrad_sq(solver_->ApplyUpdate(i, handle, rate, !clip_grads, clear_grads));
          }
          au_ids.erase(au_ids.find(id_from), au_ids.end());
        }
      }
    }
    if (param_id == END_OF_ITERATION) {
      CHECK(au_ids.empty());
      rate = -1.F;
      solver_->iteration_complete_signal(type_id);
    } else {
      au_ids.emplace(param_id);
    }
  }
  DLOG(INFO) << "[" << Caffe::current_device()
             << "] Leaving ReduceAndUpdate thread " << lwp_id();
}

void Net::add_wgrad_sq(float wgrad_sq) {
  if (wgrad_sq > 0.F) {
    wgrad_sq_.fetch_add(std::llround(wgrad_sq * GRAD_FACTOR));
  }
}

float Net::wgrad_sq() {
  return wgrad_sq_.exchange(0LL) / GRAD_FACTOR;
}

void Net::update_grad_scale() {
  global_grad_scale_coeff_ = 1.F;
  if (global_grad_scale_enabled()) {
    if (global_grad_scale_adaptive_) {
      const float wgsq = wgrad_sq();
      if (wgsq > 0.F) {
        global_grad_scale_coeff_ = std::sqrt(wgsq) * global_grad_scale_param_;
        return;
      }
    }
    global_grad_scale_coeff_ = global_grad_scale_param_;
  }
}

void Net::Reduce(int type_id, int param_id) {
  Solver::Callback* cb = solver_->callback();
  cb->reduce_barrier(type_id);
  {
    unique_ptr<unique_lock<shared_mutex>> lock;
    if (solver_->is_root()) {
      lock.reset(new unique_lock<shared_mutex>(GPUMemory::read_write_mutex()));
    }
    cb->reduce_barrier(type_id);
    cb->allreduce(type_id, param_id);
    cb->reduce_barrier(type_id);
  }
  this->learnable_params()[param_id]->
      scale_diff(1.F / (Caffe::solver_count() * global_grad_scale()),
      Caffe::cublas_handle(type_id));
  // Also need to barrier to make sure lock isn't undone
  // until all have completed, but the current nature of
  // NCCL makes this unnecessary.
  // solver_->callback()->reduce_barrier();
}

void Net::ReduceBucket(int type_id, size_t count, Type bucket_type, void* bucket) {
  Solver::Callback* cb = solver_->callback();
  cb->reduce_barrier(type_id);
  {
    unique_ptr<unique_lock<shared_mutex>> lock;
    if (solver_->is_root()) {
      lock.reset(new unique_lock<shared_mutex>(GPUMemory::read_write_mutex()));
    }
    cb->reduce_barrier(type_id);
    cb->allreduce_bucket(type_id, count, bucket, bucket_type);
    cb->reduce_barrier(type_id);
  }
  Tensor::gpu_scal(count, bucket_type, bucket, 1.F / (Caffe::solver_count() * global_grad_scale()),
      Caffe::cublas_handle(type_id));
}

void Net::ForwardDebugInfo(const int layer_id) {
  LOG_IF(INFO, Caffe::root_solver())
      << "[Forward] Layer " << layer_names_[layer_id];
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const double data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << " -> top blob " << blob_name
        << ", count: " << blob.count()
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const double data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << " -> param blob " << blob_name
        << ", count: " << blob.count()
        << " data: " << data_abs_val_mean;
  }
}

void Net::BackwardDebugInfo(const int layer_id) {
  LOG_IF(INFO, Caffe::root_solver())
      << "[Backward] Layer " << layer_names_[layer_id];
  const vector<Blob*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const double diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << " -> bottom blob " << blob_name
        << ", count: " << blob.count()
        << ", diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob& blob = *layers_[layer_id]->blobs()[param_id];
    double diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << " -> param blob " << param_id
        << ", count: " << blob.count()
        << ", diff: " << diff_abs_val_mean;
  }
}

void Net::UpdateDebugInfo(const int param_id) {
  const Blob& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const double diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    double data_abs_val_mean = blob.asum_data() / blob.count();
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

void Net::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    LayerBase* source_layer = other->layers()[i].get();
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
    vector<shared_ptr<Blob> >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
  trained_layers_shared_ = true;
}

void Net::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

void Net::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

void Net::Backward(bool apply_update) {
  BackwardFromToAu(layers_.size() - 1, 0, apply_update);
  if (debug_info_) {
    float asum_data = 0.F, asum_diff = 0.F, sumsq_data = 0.F, sumsq_diff = 0.F;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const double l2norm_data = std::sqrt(sumsq_data);
    const double l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

void Net::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

void Net::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    const string& source_layer_type = source_layer.type();
    const bool ignore_shape_mismatch = ((solver_==NULL) || solver_->param().ignore_shape_mismatch());
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
    vector<shared_ptr<Blob> >& target_blobs =
        layers_[target_layer_id]->blobs();
    if (target_blobs.size() != source_layer.blobs_size()) {
      if(source_layer_type == "BatchNorm" && ignore_shape_mismatch) {
        LOG(WARNING) << "Incompatible number of blobs for layer " << source_layer_name 
            << " target(" << target_blobs.size() << ") vs source(" << source_layer.blobs_size() << ")";    
      } else {    
        CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
            << "Incompatible number of blobs for layer " << source_layer_name 
            << " target(" << target_blobs.size() << ") vs source(" << source_layer.blobs_size() << ")";    
      }
    }
    LOG(INFO) << "Copying source layer " << source_layer_name << " Type:"
              << source_layer_type << " #blobs=" << source_layer.blobs_size();
    int num_blobs_to_copy = std::min<int>(target_blobs.size(), source_layer.blobs_size());              
    // check if BN is in legacy DIGITS format?
    if (source_layer_type == "BatchNorm" && source_layer.blobs_size() == 5) {
      for (int j = 0; j < num_blobs_to_copy; ++j) {
        const bool kReshape = true;
        target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
        DLOG(INFO) << target_blobs[j]->count();
      }
      if (target_blobs[4]->count() == 1) {
        // old format: 0 - scale , 1 - bias,  2 - mean , 3 - var, 4 - reserved
        // new format: 0 - mean  , 1 - var,  2 - reserved , 3- scale, 4 - bias
        LOG(INFO) << "BN legacy DIGITS format detected ... ";
        std::swap(target_blobs[0], target_blobs[2]);
        std::swap(target_blobs[1], target_blobs[3]);
        // ==> 0 - mean , 1 -var,  2 - scale , 3 - bias; 4 - reserved
        std::swap(target_blobs[2], target_blobs[4]);
        std::swap(target_blobs[3], target_blobs[4]);
        LOG(INFO) << "BN Transforming to new format completed.";
      }
      for (int j = 0; j < target_blobs.size(); ++j) {
        DLOG(INFO) << target_blobs[j]->count();
      }
    } else {
      for (int j = 0; j < num_blobs_to_copy; ++j) {      
        if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
          shared_ptr<Blob> source_blob = Blob::create(target_blobs[j]->data_type(),
              target_blobs[j]->diff_type());
          const bool kReshape = true;
          LOG(WARNING) << "Copying from " << source_layer_name << " to " <<
            layers_[target_layer_id]->layer_param().name() <<
            " target blob " << j;
          source_blob->FromProto(source_layer.blobs(j), kReshape);

          //Shape doesn't match. Check if atleast size matches.
          if(target_blobs[j]->count() == source_blob->count() && ignore_shape_mismatch) {
            LOG(WARNING) << "During copy param " << j << " weights from layer '"
                << source_layer_name << "'; Ignoring shape mismatch and copying forcefully.  Source param shape is "
                << source_blob->shape_string() << "; target param shape is "
                << target_blobs[j]->shape_string() << ". ";
                          
            const bool kReshape = false;
            target_blobs[j]->FromProto(source_layer.blobs(j), kReshape, ignore_shape_mismatch);
          } else {
            if(ignore_shape_mismatch) {
              LOG(WARNING) << "Cannot copy param " << j << " weights from layer '"
                << source_layer_name << "'; shape mismatch.  Source param shape is "
                << source_blob->shape_string() << "; target param shape is "
                << target_blobs[j]->shape_string() << ". "
                << "To learn this layer's parameters from scratch rather than "
                << "copying from a saved net, rename the layer.";
              } else {
                LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
                  << source_layer_name << "'; shape mismatch.  Source param shape is "
                  << source_blob->shape_string() << "; target param shape is "
                  << target_blobs[j]->shape_string() << ". "
                  << "To learn this layer's parameters from scratch rather than "
                  << "copying from a saved net, rename the layer.";
             }
          }
        } else {
          //Go ahead and copy: exactly matching blobs
          const bool kReshape = false;
          target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
        }
      }
    }
  }
  CopyQuantizationRangeInLayers();    
}

void Net::CopyTrainedLayersFrom(const string trained_filename) {
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

void Net::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

void Net::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
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
    vector<shared_ptr<Blob> >& target_blobs =
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

void Net::ToProto(NetParameter* param, bool write_diff, bool write_data) const {
  param->Clear();

  // Add bottom and top
  if(net_param_.has_quantize()) {
      param->set_name(name_ + "___QUANTIZED__SEE_END_OFTHE_FILE");
      param->set_quantize(net_param_.quantize());
  } else {
      param->set_name(name_);
  }

  if(net_param_.has_net_quantization_param()) {
      *param->mutable_net_quantization_param() = net_param_.net_quantization_param();
  }
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff, write_data);
  }
}

template<typename Dtype>
void Net::Convert2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width, int fl, bool is_unsigned, bool clip) const {
  for (int index = 0; index < cnt; ++index) {
    data[index] = data[index] * powf(2, fl);
    // Saturate data
#if CLIP_QUANT
      if(clip) {
          int qrange = is_unsigned? bit_width :  (bit_width - 1);
          Dtype max_data = +(powf(2, qrange) - 1);
          Dtype min_data = is_unsigned? 0 : -(powf(2, qrange));
          data[index] = std::max(std::min(data[index], max_data), min_data);
      }
#endif
    data[index] = round(data[index]);
    //data[index] = data[index] * pow(2, -fl);
  }
}

void Net::ToHDF5(const string& filename, bool write_diff) const {
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
        hdf5_save_nd_dataset(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset(layer_diff_hid, dataset_name.str(),
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

void Net::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

void Net::ClearParamDiffs() {
  if (Caffe::mode() == Caffe::GPU) {
    caffe_gpu_memset(learnable_space_[0].size(), 0, learnable_space_[0].data());
    caffe_gpu_memset(learnable_space_[1].size(), 0, learnable_space_[1].data());
  } else {
    for (int i = 0; i < learnable_params_.size(); ++i) {
      learnable_params_[i]->set_diff(0.F);
    }
  }
}

void Net::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) {
      gpu_prm_memory_data_use_ += params_[i]->gpu_memory_data_use();
      gpu_prm_memory_diff_use_ += params_[i]->gpu_memory_diff_use();
      continue;
    }
//    DLOG(INFO) << "param " << i << " has owner " << param_owners_[i];
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
    gpu_shp_memory_data_use_ += params_[i]->gpu_memory_data_use();
    gpu_shp_memory_diff_use_ += params_[i]->gpu_memory_diff_use();
  }
}

bool Net::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

const shared_ptr<Blob> Net::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob> blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

bool Net::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

const shared_ptr<LayerBase> Net::layer_by_name(
    const string& layer_name) const {
  shared_ptr<LayerBase> layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

void Net::set_solver(Solver* s) {
  solver_ = s;
  for (auto& layer : layers_) {
    layer->set_parent_net(this);
  }
}

void Net::InitializeLearnableDiffSpace(int type_id) {
  CHECK_GE(type_id, 0);
  CHECK_LT(type_id, 2);
  const Type t = (Type) learnable_types_[type_id];
  if (learnable_params_ptrs_[type_id].size() == learnable_params_.size()) {
    LOG(INFO) << print_current_device() << " Already reserved "
              << learnable_space_size_[type_id] << " bytes of shared learnable space for type "
              << Type_Name(t);
    return;
  }
  learnable_space_size_[type_id] = 0UL;
  learnable_params_ptrs_[type_id].resize(learnable_params_.size(), nullptr);
  for (int i = 0; i < layers_.size(); ++i) {
    for (int j = 0; j < layers_[i]->blobs().size(); ++j) {
      if (!layers_[i]->skip_apply_update(j)) {
        const int lip = layer_index_params_[make_pair(i, j)];
        if (param_owners_[lip] < 0) {
          const int param_id = learnable_param_ids_[lip];
          if (learnable_params_[param_id]->diff_type() == t) {
            learnable_space_size_[type_id] += lp_aligned_count(param_id) * lp_size(param_id);
          }
        }
      }
    }
  }
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters. Times two.
  if (learnable_space_size_[type_id] < 2) {
    learnable_space_size_[type_id] = 2;
  }
  LOG(INFO) << print_current_device() << " Reserving "
            << learnable_space_size_[type_id] << " bytes of shared learnable space for type "
            << Type_Name(t);
  learnable_space_[type_id].reserve(learnable_space_size_[type_id]);
  unsigned char* ptr = reinterpret_cast<unsigned char*>(learnable_space_[type_id].data());
  caffe_gpu_memset(learnable_space_size_[type_id], 0, ptr);
  for (int i = 0; i < layers_.size(); ++i) {
    for (int j = 0; j < layers_[i]->blobs().size(); ++j) {
      if (!layers_[i]->skip_apply_update(j)) {
        const int lip = layer_index_params_[make_pair(i, j)];
        if (param_owners_[lip] < 0) {
          const int param_id = learnable_param_ids_[lip];
          if (learnable_params_[param_id]->diff_type() == t) {
            learnable_params_[param_id]->set_gpu_diff(ptr);
            learnable_params_ptrs_[type_id][param_id] = static_cast<void*>(ptr);
            ptr += lp_aligned_count(param_id) * lp_size(param_id);
            learnable_params_mapped_.push_back(learnable_params_[param_id]);
            ltop_[type_id][i].insert(param_id);
            void *p = learnable_params_[param_id]->
                current_mutable_data_memory(Caffe::mode() == Caffe::GPU);
            (void) p;
          }
        }
      } else {
        DLOG(INFO) << print_current_device()
            << "** Skipping non-learnable blob from " << layers_[i]->name()
            << " of type " << layers_[i]->type();
      }
    }
  }
}

const vector<Type>& Net::learnable_types(bool reset) {
  if (reset || learnable_types_.empty()) {
    learnable_types_.clear();
    int type0 = -1;
    int type1 = -1;
    for (shared_ptr<Blob> lp : learnable_params_) {
      Type t = lp->diff_type();
      if (type0 < 0) {
        type0 = (int) t;
        learnable_types_.push_back(t);
      } else if (type1 < 0 && type0 != (int) t) {
        type1 = (int) t;
        learnable_types_.push_back(t);
      }
    }
    if (learnable_types_.empty() && solver_ != nullptr) {
      learnable_types_.push_back(solver_->data_type());
    }
    CHECK_LE(learnable_types_.size(), 2);
  }
  return learnable_types_;
}

template <typename Dtype>
void Net::OptimizeNet() {
  auto set_blob_data_at = [&](shared_ptr<Blob>& blob, const int n, const int c, const int h, const int w, const Dtype& value) {
    if(blob != NULL && blob->count() > 0) {
      Dtype* data = blob->mutable_cpu_data<Dtype>();
      int idx = blob->offset(n, c, h, w);
      data[idx] = value;
    }
  };
  
  auto set_blob_data_at_chan = [&](shared_ptr<Blob>& blob, const int c, const Dtype& value) {
    if(blob != NULL && blob->count() > 0) {  
      Dtype* data = blob->mutable_cpu_data<Dtype>();  
      int idx = blob->shape().size()>1 && blob->shape(0)==1? blob->offset(0,c,0,0): blob->offset(c);
      data[idx] = value;
    }
  };
    
  enum LayerSequenceType {
      LAYER_SEQ_TYPE_OTHER,
      LAYER_SEQ_TYPE_CONV_BN,
      LAYER_SEQ_TYPE_CONV_BN_SCALE,
      LAYER_SEQ_TYPE_BN_CONV,
      LAYER_SEQ_TYPE_BN_SCALE_CONV
  };

  auto layer_sequence_type = [&](int layer_id) {
    std::string layer_name = layers_[layer_id]->layer_param().name();
    std::string layer_type = layers_[layer_id]->layer_param().type();
    std::string layer_type_next = GetTopLayerType(layer_id);
    std::string layer_type_next2 = GetTopLayerType2(layer_id);
    std::string layer_type_prev = GetBottomLayerType(layer_id);
    std::string layer_type_prev2 = GetBottomLayerType2(layer_id);

    LayerSequenceType type = LAYER_SEQ_TYPE_OTHER;
    if ((layer_id < (layers_.size()-2)) && layer_type == std::string("Convolution") &&
        layer_type_next == std::string("BatchNorm") && layer_type_next2 == std::string("Scale")) {
        type = LAYER_SEQ_TYPE_CONV_BN_SCALE;
    } else if ((layer_id < (layers_.size()-1)) && layer_type == std::string("Convolution") &&
        layer_type_next == std::string("BatchNorm")) {
        type = LAYER_SEQ_TYPE_CONV_BN;
    } else if ((layer_id < (layers_.size()-1)) && layer_type == std::string("BatchNorm") &&
        layer_type_next == std::string("Convolution")) {
        type = LAYER_SEQ_TYPE_BN_CONV;
    } else if ((layer_id < (layers_.size()-2)) && layer_type == std::string("BatchNorm") &&
        layer_type_next == std::string("Scale") && layer_type_next2 == std::string("Convolution")) {
        if(layer_id>0 && layer_type_prev != std::string("Convolution")) {
          type = LAYER_SEQ_TYPE_BN_SCALE_CONV;
        }
    }
    return type;
  };


  for (int i = 0; i < (layers_.size()-1); i++) {
    LayerSequenceType layer_seq_type = layer_sequence_type(i);
    if (layer_seq_type == LAYER_SEQ_TYPE_CONV_BN || layer_seq_type == LAYER_SEQ_TYPE_CONV_BN_SCALE) {
      LOG(INFO) << "Optimizing layer: " << layers_[i]->type() << " " << layers_[i]->name();
      LayerBase& conv_layer = *layers_[i];
      //int num_groups = conv_layer.layer_param().convolution_param().group();

      // Set bias term if it not there, as it is needed when combining BN
      if(conv_layer.blobs().size()==1) {
        shared_ptr<Blob> conv_weights = conv_layer.blobs()[0];
        int channels = (conv_weights->num_axes() == 1)? conv_weights->count() : conv_weights->shape(0);
        int outputs = channels;

        bool bias_term = true;
        conv_layer.mutable_layer_param().mutable_convolution_param()->set_bias_term(bias_term);
        conv_layer.mutable_layer_param().mutable_convolution_param()->mutable_bias_filler()->set_type("constant");
        conv_layer.mutable_layer_param().mutable_convolution_param()->mutable_bias_filler()->set_value(0);

        //TODO: Revisit if needed
        conv_layer.blobs().resize(2);
        vector<int> bias_shape(1, outputs);
        conv_layer.blobs()[1] = Blob::create<Dtype>(bias_shape);
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
            conv_layer.layer_param().convolution_param().bias_filler()));
        bias_filler->Fill(conv_layer.blobs()[1].get());
      }

      shared_ptr<Blob>& conv_weights = conv_layer.blobs()[0];
      int channels = (conv_weights->num_axes() == 1)? conv_weights->count() : conv_weights->shape(0);
      //int outputs = channels;

      LayerBase& batch_norm_layer = *layers_[i+1];
      bool scale_bias = (batch_norm_layer.blobs().size() == 5);//layers_[i+1]->layer_param().batch_norm_param().scale_bias();
      bool has_scale_layer = (layer_seq_type == LAYER_SEQ_TYPE_CONV_BN_SCALE);
      shared_ptr<LayerBase> scale_layer = has_scale_layer? layers_[i+2] : NULL;
      bool has_scale_layer_bias = (has_scale_layer && scale_layer->blobs().size()>1);

      shared_ptr<Blob>& batch_norm_mean = batch_norm_layer.blobs()[0];
      shared_ptr<Blob>& batch_norm_var = batch_norm_layer.blobs()[1];
      double eps = batch_norm_layer.layer_param().batch_norm_param().eps();

      // Absorb the BatchNorm into convolution
      for(int no=0; no<conv_weights->shape(0); no++) {
        double var = batch_norm_var->data_at(no) + eps;
        double stdev_inv = std::pow(var, double(-0.5));
        double scale = scale_bias? batch_norm_layer.blobs()[3]->data_at(no) :
                (has_scale_layer? scale_layer->blobs()[0]->data_at(no) : 1.0);
        for(int ni=0; ni<conv_weights->shape(1); ni++) {
          for(int w=0; w<conv_weights->shape(2); w++) {
            for(int h=0; h<conv_weights->shape(3); h++) {
                double weights = conv_weights->data_at(no,ni,w,h);
                weights = (weights * stdev_inv * scale);
                set_blob_data_at(conv_weights, no, ni, w, h, weights);
            }
          }
        }
      }

      shared_ptr<Blob>& conv_bias = conv_layer.blobs()[1];
      for(int no=0; no<channels; no++) {
        double var = batch_norm_var->data_at(no) + eps;
        double stdev_inv = std::pow(var, double(-0.5));
        double scale = scale_bias? batch_norm_layer.blobs()[3]->data_at(no) :
                (has_scale_layer? scale_layer->blobs()[0]->data_at(no) : 1.0);
        double bias = scale_bias? batch_norm_layer.blobs()[4]->data_at(no) :
                (has_scale_layer && has_scale_layer_bias? scale_layer->blobs()[1]->data_at(no) : 0.0);
        double mean = batch_norm_mean->data_at(no);
        double weights_bias = conv_bias->data_at(no);
        weights_bias = ((weights_bias - mean) * stdev_inv * scale + bias);
        set_blob_data_at_chan(conv_bias, no, weights_bias);
      }

      // Set the batch norm (and subsequent scale layer if present) to identity
      for(int no=0; no<channels; no++) {
        if(scale_bias) {
          set_blob_data_at_chan(batch_norm_layer.blobs()[3], no, Dtype(1.0));
          set_blob_data_at_chan(batch_norm_layer.blobs()[4], no, Dtype(0.0));
        }
        if(has_scale_layer) {
          set_blob_data_at_chan(scale_layer->blobs()[0], no, Dtype(1.0));
          if(has_scale_layer_bias) {
            set_blob_data_at_chan(scale_layer->blobs()[1], no, Dtype(0.0));
          }
        }
        set_blob_data_at_chan(batch_norm_mean, no, Dtype(0.0));
        //Change var so that after adding eps, it becomes 1.0
        set_blob_data_at_chan(batch_norm_var, no, Dtype(1.0 - eps));
      }

      //transfer back to gpu
      for(int blob_id=0; blob_id<conv_layer.blobs().size(); blob_id++) {
          conv_layer.blobs()[blob_id]->gpu_data<Dtype>();
      }
      for(int blob_id=0; blob_id<batch_norm_layer.blobs().size(); blob_id++) {
          batch_norm_layer.blobs()[blob_id]->gpu_data<Dtype>();
      }
      if(has_scale_layer) {
          for(int blob_id=0; blob_id<scale_layer->blobs().size(); blob_id++) {
              scale_layer->blobs()[blob_id]->gpu_data<Dtype>();
          }
      }
    }
  }


#if 0
  //Merge a BatchNorm layer that comes before convolution layer
  //This is not needed now. BN can be run separately if it cannot be merged
  for (int i = 0; i < (layers_.size()-1); i++) {
    LayerSequenceType layer_seq_type = layer_sequence_type(i);
    if (layer_seq_type == LAYER_SEQ_TYPE_BN_CONV || layer_seq_type == LAYER_SEQ_TYPE_BN_SCALE_CONV) {
      LayerBase& batch_norm_layer = *layers_[i];
      LayerBase& conv_layer = (layer_seq_type == LAYER_SEQ_TYPE_BN_SCALE_CONV)? *layers_[i+2] : *layers_[i+1];
      shared_ptr<Blob>& conv_weights = conv_layer.blobs()[0];
      shared_ptr<Blob>& conv_bias = conv_layer.blobs()[1];
      //int channels = (conv_weights->num_axes() == 1)? conv_weights->count() : conv_weights->shape(0);
      int bn_channels = conv_weights->shape(1);
      LOG(INFO) << "Optimizing layer: " << layers_[i]->type() << " " << layers_[i]->name();
      //int num_groups = conv_layer.layer_param().convolution_param().group();

      // Set bias term if it not there, as it is needed when combining BN
      if(conv_layer.blobs().size()==1) {
        shared_ptr<Blob> conv_weights = conv_layer.blobs()[0];
        int channels = (conv_weights->num_axes() == 1)? conv_weights->count() : conv_weights->shape(0);
        int outputs = channels;

        bool bias_term = true;
        conv_layer.mutable_layer_param().mutable_convolution_param()->set_bias_term(bias_term);
        conv_layer.mutable_layer_param().mutable_convolution_param()->mutable_bias_filler()->set_type("constant");
        conv_layer.mutable_layer_param().mutable_convolution_param()->mutable_bias_filler()->set_value(0);

        //TODO: Revisit if needed
        conv_layer.blobs().resize(2);
        vector<int> bias_shape(1, outputs);
        conv_layer.blobs()[1] = Blob::create<Dtype>(bias_shape);
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
            conv_layer.layer_param().convolution_param().bias_filler()));
        bias_filler->Fill(conv_layer.blobs()[1].get());
      }

      bool scale_bias = layers_[i]->layer_param().batch_norm_param().scale_bias();
      bool has_scale_layer = (layer_seq_type == LAYER_SEQ_TYPE_BN_SCALE_CONV);
      shared_ptr<LayerBase> scale_layer = has_scale_layer? layers_[i+1] : NULL;
      bool has_scale_layer_bias = (has_scale_layer && scale_layer->blobs().size()>1);

      shared_ptr<Blob>& batch_norm_mean = batch_norm_layer.blobs()[0];
      shared_ptr<Blob>& batch_norm_var = batch_norm_layer.blobs()[1];

      Dtype eps = batch_norm_layer.layer_param().batch_norm_param().eps();

      // Absorb the BatchNorm into convolution
      for(int no=0; no<conv_weights->shape(0); no++) {
        Dtype bias_sum1 = 0;
        Dtype bias_sum2 = 0;
        for(int ni=0; ni<bn_channels; ni++) {
          Dtype var = batch_norm_var->data_at(ni) + eps;
          Dtype stdev_inv = std::pow(var, Dtype(-0.5));
          Dtype scale = scale_bias? batch_norm_layer.blobs()[3]->data_at(ni) : (has_scale_layer? scale_layer->blobs()[0]->data_at(ni) : 1.0);
          Dtype bias = scale_bias? batch_norm_layer.blobs()[4]->data_at(ni) : (has_scale_layer && has_scale_layer_bias? scale_layer->blobs()[1]->data_at(ni) : 0.0);
          Dtype mean = batch_norm_mean->data_at(ni);

          for(int w=0; w<conv_weights->shape(2); w++) {
            for(int h=0; h<conv_weights->shape(3); h++) {
              bias_sum1 += bias * conv_weights->data_at(no,ni,w,h);
              bias_sum2 += mean * stdev_inv * conv_weights->data_at(no,ni,w,h);
              set_blob_data_at(conv_weights,no,ni,w,h, conv_weights->data_at(no,ni,w,h) * stdev_inv * scale);
            }
          }
        }
        set_blob_data_at_chan(conv_bias,no, conv_bias->data_at(no) + bias_sum1 - bias_sum2);
      }

      // Set the batch norm to identity
      for(int ni=0; ni<bn_channels; ni++) {
        if(scale_bias) {      
          set_blob_data_at_chan(batch_norm_layer.blobs()[3], ni, Dtype(1.0));
          set_blob_data_at_chan(batch_norm_layer.blobs()[4], ni, Dtype(0.0));
        } else if(has_scale_layer) {
          set_blob_data_at_chan(scale_layer->blobs()[0], ni, Dtype(1.0));
          if(has_scale_layer_bias) {
            set_blob_data_at_chan(scale_layer->blobs()[1], ni, Dtype(0.0));
          }
        }
        set_blob_data_at_chan(batch_norm_mean, ni, Dtype(0.0));
        //Change var so that after adding eps, it becomes 1.0
        set_blob_data_at_chan(batch_norm_var, ni, Dtype(1.0 - eps));
      }

      //transfer back to gpu
      for(int blob_id=0; blob_id<conv_layer.blobs().size(); blob_id++) {
          conv_layer.blobs()[blob_id]->gpu_data<Dtype>();
      }
      for(int blob_id=0; blob_id<batch_norm_layer.blobs().size(); blob_id++) {
          batch_norm_layer.blobs()[blob_id]->gpu_data<Dtype>();
      }
      if(has_scale_layer) {
          for(int blob_id=0; blob_id<scale_layer->blobs().size(); blob_id++) {
              scale_layer->blobs()[blob_id]->gpu_data<Dtype>();
          }
      }

    }
  }
#endif

}


template void Net::OptimizeNet<float>();


void Net::StartQuantization() {
  bool quantize = (net_param_.quantize() && net_param_.net_quantization_param().quantization_start() > 0);
  if(quantize) {
    const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();
    if(infer_count_ >= net_qparam.quantization_start()) {
      if(infer_count_ == net_qparam.quantization_start()) {
        LOG(INFO)<< "Enabling quantization flag in quantization_param at infer/iter index: " << infer_count_;
        this->EnableQuantizationForSelectedLayers();
      }
      this->SetQuantizationParams();
    }
  }
}

void Net::FinishQuantization() {
  bool quantize = (net_param_.quantize() && net_param_.net_quantization_param().quantization_start() > 0);
  if(quantize) {
    const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();
    if(net_qparam.update_quantization_param()) {
        this->UpdateQuantizationRangeInLayers();
    }

    if(net_qparam.quantization_start() > 0 && infer_count_ >= net_qparam.quantization_start()) {
      string phase = this->phase() == caffe::TRAIN ? "Train" : "Test";
      if (net_qparam.display_quantization() > 0 && (infer_count_ % net_qparam.display_quantization() == 0)) {
        LOG(INFO)<< "Quantizing the net: " << this->name() + " " + phase;
        this->DisplayQuantizationParams();
      }
    }
  }
}

void Net::ClearQuantizationRangeInLayers() {
  max_in_.clear();
  max_out_.clear();
  max_weights_.clear();

  min_in_.clear();
  min_out_.clear();
  min_weights_.clear();
}

void Net::CopyQuantizationRangeInLayers() {
  max_in_.resize(layers_.size());
  max_out_.resize(layers_.size());
  max_weights_.resize(layers_.size());

  min_in_.resize(layers_.size());
  min_out_.resize(layers_.size());
  min_weights_.resize(layers_.size());

  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    min_in_[layer_id].resize(bottom_vecs_[layer_id].size(), 0);
    max_in_[layer_id].resize(bottom_vecs_[layer_id].size(), 0);
    min_weights_[layer_id].resize(layers_[layer_id]->blobs().size(), 0);
    max_weights_[layer_id].resize(layers_[layer_id]->blobs().size(), 0);
    min_out_[layer_id].resize(top_vecs_[layer_id].size(), 0);
    max_out_[layer_id].resize(top_vecs_[layer_id].size(), 0);
  }

  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    if(!layers_[layer_id]->layer_param().has_quantization_param()) {
      continue;
    }
    const QuantizationParameter& source_quantization_param = layers_[layer_id]->layer_param().quantization_param();
    for(int blob_id = 0; blob_id<min_in_[layer_id].size(); blob_id++) {
      min_in_[layer_id][blob_id] = source_quantization_param.qparam_in(blob_id).min();
      max_in_[layer_id][blob_id] = source_quantization_param.qparam_in(blob_id).max();
    }
    for(int blob_id = 0; blob_id<layers_[layer_id]->blobs().size(); blob_id++) {
      min_weights_[layer_id][blob_id] = source_quantization_param.qparam_w(blob_id).min();
      max_weights_[layer_id][blob_id] = source_quantization_param.qparam_w(blob_id).max();
    }
    for(int blob_id = 0; blob_id<min_out_[layer_id].size(); blob_id++) {
      min_out_[layer_id][blob_id] = source_quantization_param.qparam_out(blob_id).min();
      max_out_[layer_id][blob_id] = source_quantization_param.qparam_out(blob_id).max();
    }
  }
}

void Net::UpdateQuantizationRangeInLayers() {
  const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();

  max_in_.resize(layers_.size());
  max_out_.resize(layers_.size());
  max_weights_.resize(layers_.size());

  min_in_.resize(layers_.size());
  min_out_.resize(layers_.size());
  min_weights_.resize(layers_.size());

  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    min_in_[layer_id].resize(bottom_vecs_[layer_id].size(), 0);
    max_in_[layer_id].resize(bottom_vecs_[layer_id].size(), 0);
    min_weights_[layer_id].resize(layers_[layer_id]->blobs().size(), 0);
    max_weights_[layer_id].resize(layers_[layer_id]->blobs().size(), 0);
    min_out_[layer_id].resize(top_vecs_[layer_id].size(), 0);
    max_out_[layer_id].resize(top_vecs_[layer_id].size(), 0);
  }

  // Find maximal values.
  float range_expansion_factor = net_qparam.range_expansion_factor();
  float beta = (infer_count_ == 0? 1.0 : net_qparam.range_update_factor());
  float alpha = (1.0 - beta);

  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    if(bottom_vecs_[layer_id].size()>0) {
        for(int blob_id = 0; blob_id<bottom_vecs_[layer_id].size(); blob_id++) {
            float min_in = bottom_vecs_[layer_id][blob_id]->min(0, 0);
            float max_in = bottom_vecs_[layer_id][blob_id]->max(0, 0);
            min_in *= range_expansion_factor;
            max_in *= range_expansion_factor;
            min_in_[layer_id][blob_id] = min_in_[layer_id][blob_id] * alpha +  min_in * beta;
            max_in_[layer_id][blob_id] = max_in_[layer_id][blob_id] * alpha +  max_in * beta;
        }
    }

    if(top_vecs_[layer_id].size()>0) {
        for(int blob_id = 0; blob_id<top_vecs_[layer_id].size(); blob_id++) {
            float min_out = top_vecs_[layer_id][blob_id]->min(0, 0);
            float max_out = top_vecs_[layer_id][blob_id]->max(0, 0);
            min_out *= range_expansion_factor;
            max_out *= range_expansion_factor;
            min_out_[layer_id][blob_id] = min_out_[layer_id][blob_id] * alpha +  min_out * beta;
            max_out_[layer_id][blob_id] = max_out_[layer_id][blob_id] * alpha +  max_out * beta;
        }
    }

    //TODO: Set to 1 to consider the weights only, and ignore the bias
    int max_params_to_consider = INT_MAX;
    int num_params = std::min((int)layers_[layer_id]->blobs().size(), max_params_to_consider);
    if(num_params > 0) {
        for(int blob_id = 0; blob_id < num_params; blob_id++) {
          float min_weights = (float)layers_[layer_id]->blobs()[blob_id]->min(0, 0);
          float max_weights = (float)layers_[layer_id]->blobs()[blob_id]->max(0, 0);
          //for weights, we can use the actual range - no need for running average.
          //min_weights *= expansion_factor;
          //max_weights *= expansion_factor;
          //min_weights_[layer_id] = min_weights_[layer_id] * alpha + min_weights * beta;
          //max_weights_[layer_id] = max_weights_[layer_id] * alpha + max_weights * beta;
          min_weights_[layer_id][blob_id] = min_weights;
          max_weights_[layer_id][blob_id] = max_weights;
        }
    }
  }
}

vector<int> Net::GetTopLayerIds(int layer_id, bool only_one) {
    vector<int> top_layer_ids;
    vector<int> cur_top_ids = this->top_ids(layer_id);
    for(int t=0; t<cur_top_ids.size(); t++) {
      int tid = cur_top_ids[t];
      int found_lid = -1;
      for(int search_lid=(layer_id+1); search_lid<layers_.size(); search_lid++) {
        vector<int> seach_bids = this->bottom_ids(search_lid);
        for(int b=0; b<seach_bids.size(); b++) {
            if(tid == seach_bids[b]) {
                found_lid = search_lid;
                break;
            }
        }
        if(found_lid >= 0) {
            top_layer_ids.push_back(found_lid);
            if(only_one) {
                break;
            }
        }
      }
    }
    return top_layer_ids;
}
vector<int> Net::GetBottomLayerIds(int layer_id, bool only_one) {
    vector<int> bottom_layer_ids;
    vector<int> cur_bottom_ids = this->bottom_ids(layer_id);
    for(int b=0; b<cur_bottom_ids.size(); b++) {
      int bid = cur_bottom_ids[b];
      int found_lid = -1;
      for(int search_lid=0; search_lid<layer_id; search_lid++) {
        vector<int> seach_tids = this->top_ids(search_lid);
        for(int t=0; t<seach_tids.size(); t++) {
            if(search_lid < layer_id && bid == seach_tids[t]) {
                found_lid = search_lid;
            }
        }
      }
      if(found_lid >= 0) {
          bottom_layer_ids.push_back(found_lid);
          if(only_one) {
              break;
          }
      }
    }
    return bottom_layer_ids;
}
string Net::GetTopLayerType(int layer_id, bool only_one) {
    vector<int> layer_ids = GetTopLayerIds(layer_id);
    string layer_types = "";
    for(int i=0; i<layer_ids.size(); i++) {
        if(layer_types != "") {
            layer_types += ",";
        }
        layer_types += layers_[layer_ids[i]]->type();
        if(only_one) {
            break;
        }
    }
    return layer_types;
}
string Net::GetBottomLayerType(int layer_id, bool only_one) {
    vector<int> layer_ids = GetBottomLayerIds(layer_id);
    string layer_types = "";
    for(int i=0; i<layer_ids.size(); i++) {
        if(layer_types != "") {
            layer_types += ",";
        }
        layer_types += layers_[layer_ids[i]]->type();
        if(only_one) {
            break;
        }
    }
    return layer_types;
}
string Net::GetTopLayerType2(int layer_id, bool only_one) {
    vector<int> layer_ids = GetTopLayerIds(layer_id);
    string layer_types = "";
    if(layer_ids.size()==1) {
        int top_layer_id = layer_ids[0];
        vector<int> layer_ids2 = GetTopLayerIds(top_layer_id);
        for(int i=0; i<layer_ids2.size(); i++) {
            if(layer_types != "") {
                layer_types += ",";
            }
            layer_types += layers_[layer_ids2[i]]->type();
            if(only_one) {
                break;
            }
        }
        return layer_types;
    } else {
        return "";
    }
}
string Net::GetBottomLayerType2(int layer_id, bool only_one) {
    vector<int> layer_ids = GetBottomLayerIds(layer_id);
    string layer_types = "";
    if(layer_ids.size()==1) {
        int bottom_layer_id = layer_ids[0];
        vector<int> layer_ids2 = GetBottomLayerIds(bottom_layer_id);
        for(int i=0; i<layer_ids2.size(); i++) {
            if(layer_types != "") {
                layer_types += ",";
            }
            layer_types += layers_[layer_ids2[i]]->type();
            if(only_one) {
                break;
            }
        }
        return layer_types;
    } else {
        return "";
    }
}
vector<const QuantizationParameter::QParams*> Net::GetBottomLayerQParams(int layer_id) {
    vector<const QuantizationParameter::QParams*> bottom_qparams;
    for(int b=0; b<this->bottom_ids(layer_id).size(); b++) {
        int search_bid = this->bottom_ids(layer_id)[b];
        int found_lid = -1;
        //int found_tid = -1;
        int found_top = -1;
        for (int search_lid = 0; search_lid < layer_id; search_lid++) {
            for(int t=0; t<this->top_ids(search_lid).size(); t++) {
              int search_tid = this->top_ids(search_lid)[t];
              if(search_bid == search_tid) {
                  found_lid = search_lid;
                  //found_tid = search_tid;
                  found_top = t;
              }
            }
        }
        if(found_lid >= 0) {
            const QuantizationParameter::QParams& qparam_out = layers_[found_lid]->layer_param().quantization_param().qparam_out(found_top);
            bottom_qparams.push_back(&qparam_out);
        }
    }
    return bottom_qparams;
}

void Net::EnableQuantizationForSelectedLayers() {
  const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();
  if(net_qparam.update_quantization_param()) {
    for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
      std::string layer_name = layers_[layer_id]->layer_param().name();
      std::string layer_type = layers_[layer_id]->layer_param().type();

      //LOG(INFO) << "Checking whether quantization is needed for: " << layer_type << " " << layer_name;

     //It is assumed that operations across multiple blobs (such as eltwise) is done in high precision.
     //So, we don't need to align the quantization ranges of different inputs.
     //Infact we don't quantize the input - it is assumed to be already quantized when it reaches the current layer.

      //find if this is a merged layer
      std::string layer_type_next = GetTopLayerType(layer_id);
      std::string layer_type_next2 = GetTopLayerType2(layer_id);
      std::string layer_type_prev = GetBottomLayerType(layer_id);
      std::string layer_type_prev2 = GetBottomLayerType2(layer_id);

      //std::cout << "LinearLayerChain: " << "(" << layer_type_prev2 << ")(" << layer_type_prev << ")("
      //        << layer_type << " - " << layer_name << ")" << "(" << layer_type_next << ")("
      //        << layer_type_next2 << ")" << std::endl;

      bool is_ignored_layer_name = false;
      for(int i=0; i<net_qparam.ignored_layer_names_size(); i++) {
          if(layer_name == net_qparam.ignored_layer_names(i)) {
              is_ignored_layer_name = true;
          }
      }

      bool is_merged_layer = false;
      if(layer_type == "Convolution" || layer_type == "InnerProduct" || layer_type == "Deconvolution") {
          if(layer_type_next == "BatchNorm" || layer_type_next == "Scale" || layer_type_next == "ReLU") {
              is_merged_layer = true;
          }
      } else if(layer_type == "BatchNorm") {
          if(layer_type_next == "Scale" || layer_type_next == "ReLU") {
              is_merged_layer = true;
          } else if(layer_type_next == "Convolution") {
              if(layer_type_prev != "Convolution") {
                  is_merged_layer = true;
              }
          } else if(layer_type_next == "Convolution" || layer_type_next == "Scale") {
              if(layer_type_prev != "Convolution") {
                  is_merged_layer = true;
              }
          }
      } else if(layer_type == "Scale") {
          if(layer_type_next == "ReLU") {
              is_merged_layer = true;
          } else if(layer_type_next == "Convolution") {
              if(layer_type_prev != "Convolution" && layer_type_prev2 != "Convolution") {
                  is_merged_layer = true;
              }
          }
      }

      //for data layers, only quantize the first output and ignore the others (eg. label)
      int max_blobs_to_quantize = INT_MAX;
      string layer_type_lower = layer_type;
      std::transform(layer_type_lower.begin(), layer_type_lower.end(), layer_type_lower.begin(),
              [](unsigned char c) {return std::tolower(c);}
      );
      if(layer_type_lower.find("data") != string::npos) {
          max_blobs_to_quantize = 1;
      }

      bool is_quantized_layer_type = false;
      if(layer_type == "Convolution" || layer_type == "InnerProduct" || layer_type == "Deconvolution" ||
              layer_type == "BatchNorm" || layer_type == "Scale" || layer_type == "ReLU" ||
              layer_type == "PReLU" || layer_type == "Eltwise" || layer_type == "Concat" ||
              layer_type == "Bias" || layer_type == "Pooling") {
          is_quantized_layer_type = true;
      }
      if(layer_type_lower.find("data") != string::npos) {
          is_quantized_layer_type = true;
      }

      //quantize weights
      if(net_qparam.quantize_weights()) {
          if(is_quantized_layer_type && (!is_merged_layer) && (!is_ignored_layer_name)) {
              if(layer_type == "Convolution" || layer_type == "InnerProduct" || layer_type == "Deconvolution") {
                  QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
                  for(int blob_id=0; blob_id<layers_[layer_id]->blobs().size(); blob_id++) {
                    if(quantization_param.qparam_w_size() <= blob_id) {
                      quantization_param.add_qparam_w();
                    }
                    quantization_param.mutable_qparam_w(blob_id)->set_quantize(true);
                  }
              }
          }
      }

      //quantize output activations
      if(net_qparam.quantize_activations()) {
          if(is_quantized_layer_type && (!is_merged_layer) && (!is_ignored_layer_name)) {
              QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
              for(int blob_id=0; blob_id<top_vecs_[layer_id].size(); blob_id++) {
                if(quantization_param.qparam_out_size() <= blob_id) {
                  quantization_param.add_qparam_out();
                }
                if(blob_id < max_blobs_to_quantize) {
                  quantization_param.mutable_qparam_out(blob_id)->set_quantize(true);
                }
              }
              LOG(INFO) << "Enabling quantization at output of: " << layer_type << " " << layer_name;
          }
      }
    }
  }
}

void Net::SetQuantizationParams() {
  const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();

  if(net_qparam.update_quantization_param()) {
    //insert quantization_param in the layers that do not have it
    QuantizationParameter_Rounding rounding_scheme = (this->phase() == caffe::TRAIN ?
            QuantizationParameter_Rounding_STOCHASTIC : net_qparam.rounding_scheme());

    for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
        QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();

        quantization_param.set_precision(net_qparam.precision());
        quantization_param.set_rounding_scheme(rounding_scheme);
        quantization_param.set_power2_scale_weights(net_qparam.power2_scale_weights());
        quantization_param.set_power2_scale_activations(net_qparam.power2_scale_activations());
        quantization_param.set_quantized_infer_count(infer_count_ - net_qparam.quantization_start());

        // quantize parameters
        SetQuantizationParamsLayerWeights(layer_id);

        // quantize input activations
        SetQuantizationParamsLayerInput(layer_id);

        // quantize output activations
        SetQuantizationParamsLayerOutput(layer_id);
    }
  }

  //this->DisplayQuantizationParams();
}

int Net::EstimateAbsBits(float val) {
    return (val!=0)? ceil(log2(std::fabs(val))) : 0;
}

void Net::EstiamteQScaleParams(float min, float max, int bitwidth, bool power2_scale,
    bool unsigned_data, bool apply_offset, QuantizationParameter::QParams& qparam_xx) {
  float scale_applied = qparam_xx.scale_applied();

  qparam_xx.set_bitwidth(bitwidth);
  qparam_xx.set_unsigned_data(unsigned_data);
  qparam_xx.set_unsigned_quant(unsigned_data || apply_offset);
  qparam_xx.set_min(min);
  qparam_xx.set_max(max);

  //fractbits cannot be computed in cases where non-power of 2 quant is involved. not using it.
  qparam_xx.set_fracbits(0);

  float max_val_abs = std::max(std::fabs(max), std::fabs(min))*scale_applied;
  float max_val_range = std::abs(max - min)*scale_applied;

  if(power2_scale) {
    int estimated_bits = apply_offset? EstimateAbsBits(max_val_range) :
        (unsigned_data? EstimateAbsBits(max_val_abs) : (EstimateAbsBits(max_val_abs)+1));
    int shiftbits = estimated_bits - bitwidth;
    //account for the scaling due to right shift by shiftbits
    float scale_relative = 1.0/((shiftbits>=0)? float(1<<shiftbits) : float(1.0/(1<<std::abs(shiftbits))));
    float scale_target = scale_relative*scale_applied;

    qparam_xx.set_shiftbits(shiftbits);
    qparam_xx.set_scale_target(scale_target);
    qparam_xx.set_offset(apply_offset? (0 - min * scale_target) : 0);
  } else {
    //We can even use (1L<<bitwidth). Since we clip the quantized output - this should not be an issue.
    //However found that ((1L<<bitwidth)-1) gave slightly better quality
    float max_qrange = ((1L<<bitwidth)-1);
    float max_qrange_half = ((1L<<(bitwidth-1))-1);
    float scale_relative = apply_offset? max_qrange/max_val_range :
      (unsigned_data? max_qrange/max_val_abs : max_qrange_half/max_val_abs);
    float scale_target = scale_relative*scale_applied;

    //shiftbits are not integer - so cannot be set accurately.
    qparam_xx.set_shiftbits(0);
    qparam_xx.set_offset(apply_offset? (0 - min * scale_target) : 0);
    qparam_xx.set_scale_target(scale_target);
  }

  //the new scale target is on top of the scale applied. so the effective scale will be a product of both,
  //after the scaling is applied.
}

void Net::SetQuantizationParamsLayerInput(const int layer_id) {
  //const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();
  QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();

  vector<const QuantizationParameter::QParams*> qparam_bot_vec = this->GetBottomLayerQParams(layer_id);
  int num_bottom_vecs = bottom_vecs_[layer_id].size();
  CHECK(qparam_bot_vec.size() == num_bottom_vecs) << "Incorrect number of bottom qparams obtained";

  //It is assumed that operations across multiple blobs (such as eltwise) is done in high precision.
  //So, we don't need to align the quantization ranges of different inputs.
  for(int blob_id = 0; blob_id<num_bottom_vecs; blob_id++) {
    if(quantization_param.qparam_in_size() <= blob_id) {
      quantization_param.add_qparam_in();
    }

    //Get the scale applied to this blob
    QuantizationParameter::QParams& qparam_in = *quantization_param.mutable_qparam_in(blob_id);
    CHECK(qparam_bot_vec[blob_id] != NULL) << "Input QParams should not be NULL";
    if(qparam_bot_vec[blob_id]->has_scale_applied()) {
      qparam_in.set_scale_applied(qparam_bot_vec[blob_id]->scale_applied());
    }

    const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();
    float min_layer = min_in_[layer_id][blob_id];
    float max_layer = max_in_[layer_id][blob_id];
    bool unsigned_data = (min_layer>=0);
    EstiamteQScaleParams(min_layer, max_layer, net_qparam.bitwidth_activations(),
       net_qparam.power2_scale_activations(), unsigned_data, net_qparam.apply_offset_activations(), qparam_in);

    if(qparam_in.quantize()) {
        qparam_in.set_scale_applied(qparam_in.scale_target());
    }
  }
}

void Net::SetQuantizationParamsLayerOutput(const int layer_id) {
  const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();
  QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
  std::string layer_type = layers_[layer_id]->layer_param().type();

  int num_bottom_vecs = bottom_vecs_[layer_id].size();
  float scale_applied_in_max = num_bottom_vecs>0? -1.0: 1.0;
  //Get the max scale applied in the case of multiple blobs
  for(int blob_id = 0; blob_id<num_bottom_vecs; blob_id++) {
    QuantizationParameter::QParams& qparam_in = *quantization_param.mutable_qparam_in(blob_id);
    scale_applied_in_max = std::max(scale_applied_in_max, qparam_in.scale_applied());
  }

  //this scale will be applied only in the quantization function,
  //but we need to calculate the scaling effect (on the output) due to quantization scale of weights.
  float scale_target_w = (layer_type == "Convolution" || layer_type == "InnerProduct" || layer_type == "Deconvolution")?
          quantization_param.mutable_qparam_w(0)->scale_target() : 1.0;

  int num_top_vecs = top_vecs_[layer_id].size();
  for(int blob_id = 0; blob_id<num_top_vecs; blob_id++) {
      if(quantization_param.qparam_out_size() <= blob_id) {
        quantization_param.add_qparam_out();
      }
      float min_layer = min_out_[layer_id][blob_id];
      float max_layer = max_out_[layer_id][blob_id];
      bool unsigned_data = (min_layer>=0);

      //the convolution operations in floating point doesn't change the applied scale,
      //but since the weights are quantized, there is a scaling that happens due to it.
      //applied for it by multiplying with it.
      QuantizationParameter::QParams& qparam_out = *quantization_param.mutable_qparam_out(blob_id);
      if(layer_type == "Convolution" || layer_type == "InnerProduct" || layer_type == "Deconvolution") {
          qparam_out.set_scale_applied(scale_applied_in_max * scale_target_w);
      }

      EstiamteQScaleParams(min_layer, max_layer, net_qparam.bitwidth_activations(),
          net_qparam.power2_scale_activations(), unsigned_data, net_qparam.apply_offset_activations(), qparam_out);

      int num_blobs = layers_[layer_id]->blobs().size();
      int fracbits_in = quantization_param.qparam_in_size()>0? quantization_param.qparam_in(0).fracbits() : 0;
      int fracbits_out = qparam_out.fracbits();
      int fracbits_weights = num_blobs>0? quantization_param.qparam_w(0).fracbits() : 0;

      if(layer_type == "Convolution" || layer_type == "InnerProduct" || layer_type == "Deconvolution") {
          //special handling for bias.
          //scale factor for bias = scale_in * scale_w
          if(blob_id > 0) {
              QuantizationParameter::QParams& qparam_in = *quantization_param.mutable_qparam_out(0);
              QuantizationParameter::QParams& qparam_w = *quantization_param.mutable_qparam_w(0);
              float scale_target = qparam_in.scale_target() * qparam_w.scale_target();
              qparam_out.set_scale_target(scale_target);
          } else {
              //avoid left shift at output - will lose accuracy
              if((fracbits_in + fracbits_weights) < fracbits_out) {
                fracbits_out = (fracbits_in + fracbits_weights);
                qparam_out.set_fracbits(fracbits_out);
              }
              //if((fracbits_in + fracbits_weights) < fracbits_out) {
              //  LOG(FATAL) << "Qformat error for layer: " << layers_[layer_id]->layer_param().name()
              //      << " fracbits_in:" << fracbits_in << " fracbits_weights:" << fracbits_weights
              //      << " fracbits_out:" << fracbits_out;
              //}
          }
      }

      if(qparam_out.quantize()) {
        qparam_out.set_scale_applied(qparam_out.scale_target());
        //LOG(INFO) << layer_type << ": " << blob_id << ": " << "scale_applied:" << qparam_out.scale_applied();
      }
  }
}

void Net::SetQuantizationParamsLayerWeights(const int layer_id) {
  const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();
  QuantizationParameter& quantization_param = *layers_[layer_id]->mutable_layer_param().mutable_quantization_param();
  std::string layer_type = layers_[layer_id]->layer_param().type();

  int num_blobs = layers_[layer_id]->blobs().size();
  for(int blob_id = 0; blob_id<num_blobs; blob_id++) {
    if(quantization_param.qparam_w_size() <= blob_id) {
      quantization_param.add_qparam_w();
    }

    float min_layer = min_weights_[layer_id][blob_id];
    float max_layer = max_weights_[layer_id][blob_id];
    bool unsigned_data = (min_layer>=0);
    int bitwidth = (layer_type == "BatchNorm" || blob_id > 0)? net_qparam.bitwidth_bias() : net_qparam.bitwidth_weights();
    QuantizationParameter::QParams& qparam_w = *quantization_param.mutable_qparam_w(blob_id);
    EstiamteQScaleParams(min_layer, max_layer, bitwidth,
        net_qparam.power2_scale_weights(), unsigned_data, net_qparam.apply_offset_weights(), qparam_w);

    //if we set scale_applied for weights, the next time scale_target computation will go wrong.
    //since the same values and blob are re-used every frame. Anyway, scale_applied of weights is not used.
    //if(qparam_w.quantize()) {
    //    qparam_w.set_scale_applied(qparam_w.scale_target());
    //}
  }
}


void Net::DisplayQuantizationParams() {
  const NetQuantizationParameter& net_qparam = net_param_.net_quantization_param();

  for (int i = 0; i < layers_.size(); ++i) {
    if (layers_[i]->layer_param().has_quantization_param()) {
      // if this is a convolutional layer which should be quantized ...
      QuantizationParameter& quantization_param = *layers_[i]->mutable_layer_param().mutable_quantization_param();
      int num_blobs = layers_[i]->blobs().size();
      if (quantization_param.qparam_w_size()>0 && net_qparam.quantize_weights() && num_blobs>0 &&
              quantization_param.qparam_w(0).quantize()) {
        LOG(INFO)<<" Q weights:" << i << " Name:" << layers_[i]->layer_param().name() <<
        " bitwidth:" << quantization_param.qparam_w(0).bitwidth() <<
        " fracbits:" << quantization_param.qparam_w(0).fracbits() <<
        " scale:" << quantization_param.qparam_w(0).scale_target() <<
        " offset:" << quantization_param.qparam_w(0).offset() <<
        " unsigned_data:" << quantization_param.qparam_w(0).unsigned_data() <<
        " min:" << quantization_param.qparam_w(0).min() <<
        " max:" << quantization_param.qparam_w(0).max() <<
        " scale_applied:" << quantization_param.qparam_w(0).scale_applied() <<
        " scale_target:" << quantization_param.qparam_w(0).scale_target();
      }

      if (quantization_param.qparam_w_size()>1 && net_qparam.quantize_weights() && num_blobs>1 &&
              quantization_param.qparam_w(1).quantize()) {
        LOG(INFO)<<" Q bias:" << i << " Name:" << layers_[i]->layer_param().name() <<
        " bitwidth:" << quantization_param.qparam_w(1).bitwidth() <<
        " fracbits:" << quantization_param.qparam_w(1).fracbits() <<
        " scale:" << quantization_param.qparam_w(1).scale_target() <<
        " offset:" << quantization_param.qparam_w(1).offset() <<
        " unsigned_data:" << quantization_param.qparam_w(1).unsigned_data() <<
        " min:" << quantization_param.qparam_w(1).min() <<
        " max:" << quantization_param.qparam_w(1).max() <<
        " scale_applied:" << quantization_param.qparam_w(1).scale_applied() <<
        " scale_target:" << quantization_param.qparam_w(1).scale_target();
      }

      if (quantization_param.qparam_in_size()>0 && net_qparam.quantize_activations() &&
              quantization_param.qparam_in(0).quantize()) {
        int num_bottom_vecs = bottom_vecs_[i].size();
        std::stringstream ss;
        ss << " Q input :" << i << " Name:" << layers_[i]->layer_param().name();
        for(int blob_id=0; blob_id<std::min<int>(num_bottom_vecs, quantization_param.qparam_in_size()); blob_id++) {
          ss << " bitwidth:" << quantization_param.qparam_in(blob_id).bitwidth();
          ss << " fracbits:" << quantization_param.qparam_in(blob_id).fracbits();
          ss << " scale:" << quantization_param.qparam_in(blob_id).scale_target() ;
          ss << " offset:" << quantization_param.qparam_in(blob_id).offset() ;
          ss << " unsigned_data:" << quantization_param.qparam_in(blob_id).unsigned_data();
          ss << " min:" << quantization_param.qparam_in(blob_id).min();
          ss << " max:" << quantization_param.qparam_in(blob_id).max();
          ss << " scale_applied:" << quantization_param.qparam_in(blob_id).scale_applied();
          ss << " scale_target:" << quantization_param.qparam_in(blob_id).scale_target();
        }
        LOG(INFO) << ss.str();
      }

      if (quantization_param.qparam_out_size()>0 && net_qparam.quantize_activations() &&
              quantization_param.qparam_out(0).quantize()) {
        LOG(INFO)<< " Q output:" << i << " Name:" << layers_[i]->layer_param().name() <<
        " bitwidth:" << quantization_param.qparam_out(0).bitwidth() <<
        " fracbits:" << quantization_param.qparam_out(0).fracbits() <<
        " scale:" << quantization_param.qparam_out(0).scale_target() <<
        " offset:" << quantization_param.qparam_out(0).offset() <<
        " unsigned_data:" << quantization_param.qparam_out(0).unsigned_data() <<
        " min:" << quantization_param.qparam_out(0).min() <<
        " max:" << quantization_param.qparam_out(0).max() <<
        " scale_applied:" << quantization_param.qparam_out(0).scale_applied() <<
        " scale_target:" << quantization_param.qparam_out(0).scale_target();
      }
    }
  }
}

void Net::DisableQuantization() {
  for (int i = 0; i < layers_.size(); ++i) {
    if (layers_[i]->layer_param().has_quantization_param()) {
      QuantizationParameter& quantization_param = *layers_[i]->mutable_layer_param().mutable_quantization_param();
      quantization_param.set_precision(QuantizationParameter_Precision_FLOAT);
    }
  }
}


//Old, deprecated function.
void Net::FindAndApplyThresholdNet(float threshold_fraction_low, float threshold_fraction_mid, float threshold_fraction_high,
    float threshold_value_maxratio, float threshold_value_max, float threshold_step_factor, bool verbose) {

  for (int i = 0; i < layers_.size(); i++) {
    if (layers_[i]->type() == std::string("Convolution")) {
      LayerBase& conv_layer = *layers_[i];
      Blob& conv_weights = *conv_layer.blobs()[0];
      const ConvolutionParameter& conv_param = layers_[i]->layer_param().convolution_param();
      int num_group = layers_[i]->layer_param().convolution_param().group();
      //int stride = layers_[i]->layer_param().convolution_param().stride_size()>0? layers_[i]->layer_param().convolution_param().stride(0) : 1;

      int no = (conv_weights.num_axes() == 1)? conv_weights.count() : conv_weights.shape(0);
      int ni = ((conv_weights.num_axes() == 1)? conv_weights.count() : conv_weights.shape(1))*num_group;
      float count = conv_weights.count();
      if(verbose) {
        LOG(WARNING) << layers_[i]->layer_param().name() << " ni=" << ni << " no=" << no;
      }

      int kernel_shape_data[2];
      if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
        kernel_shape_data[0] = conv_param.kernel_h();
        kernel_shape_data[1] = conv_param.kernel_w();
      } else {
        const int num_kernel_dims = conv_param.kernel_size_size();
        for (int i = 0; i < 2; ++i) {
          kernel_shape_data[i] = conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
        }
      }

      //need to add it as cfg option :FIX_ME:SN
      const bool no_sparsity_for_small_kernel = true;
      bool need_sparsity_for_this_layer = true;
      if (no_sparsity_for_small_kernel) {
        need_sparsity_for_this_layer = (kernel_shape_data[0] > 2) && (kernel_shape_data[1] > 2) && (num_group != no);
      }

      if((ni>=32 || no >= 32) && num_group<no && need_sparsity_for_this_layer) {
        float threshold_fraction_selected = ((ni>=256 && no >= 512)? threshold_fraction_high :
            ((ni>=32 && no >= 32)? threshold_fraction_mid: threshold_fraction_low));
        float selected_threshold = 0;
        float max_abs = std::abs(conv_weights.max(0, 0));
        float min_abs = std::abs(conv_weights.min(0, 0));
        float max_abs_value = std::max<float>(max_abs, min_abs);
        float step_size = max_abs_value * threshold_step_factor;
        float max_threshold_value = std::min<float>(threshold_value_max, max_abs_value*threshold_value_maxratio);

        float step_sizeX = step_size*100;
        float selected_thresholdX = 0;
        for(float step=0; step<max_abs_value && step<max_threshold_value; step+=step_sizeX) {
          float zcount = conv_weights.count_zero((float)step, 0, 0);
          float zratio = zcount / count;
          if(zratio <= threshold_fraction_selected) {
            selected_thresholdX = step;
          } else {
            break;
          }
        }

        for(float step=std::max((selected_thresholdX-step_sizeX),0.0f);
            step<(selected_thresholdX+step_sizeX) && step<max_abs_value && step<max_threshold_value;
            step+=step_size) {
          float zcount = conv_weights.count_zero((float)step, 0, 0);
          float zratio = zcount / count;
          if(zratio <= threshold_fraction_selected) {
            selected_threshold = step;
          } else {
            break;
          }
        }

        conv_weights.zerout(selected_threshold, 0, 0);

        if(verbose) {
          float zcount = conv_weights.count_zero(0.0, 0, 0);
          LOG(WARNING) << layers_[i]->layer_param().name() << " MaxAbsWeight=" << max_abs_value
              << " MaxThreshold=" << max_threshold_value << " SelectedThreshold=" << selected_threshold
              << " ZeroPercentage=" << (zcount*100/count);
        }
      }
    }
  }
}


void Net::FindAndApplyChannelThresholdNet(float threshold_fraction_low, float threshold_fraction_mid, float threshold_fraction_high,
    float threshold_value_maxratio, float threshold_value_max, float threshold_step_factor, bool verbose) {

  for (int i = 0; i < layers_.size(); i++) {
    if (layers_[i]->type() == std::string("Convolution")) {
      LayerBase& conv_layer = *layers_[i];
      Blob& conv_weights = *conv_layer.blobs()[0];
      const ConvolutionParameter& conv_param = layers_[i]->layer_param().convolution_param();
      const string layer_name = layers_[i]->layer_param().name();

      int num_group = conv_param.group();
      //int stride = conv_param.stride_size()>0? conv_param.stride(0) : 1;
      int kernel_shape_data[2];
      if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
        kernel_shape_data[0] = conv_param.kernel_h();
        kernel_shape_data[1] = conv_param.kernel_w();
      } else {
        const int num_kernel_dims = conv_param.kernel_size_size();
        for (int i = 0; i < 2; ++i) {
          kernel_shape_data[i] = conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
        }
      }

      int no = (conv_weights.num_axes() == 1)? conv_weights.count() : conv_weights.shape(0);
      int ni = ((conv_weights.num_axes() == 1)? conv_weights.count() : conv_weights.shape(1))*num_group;
      float count = conv_weights.count();
      if(verbose) {
        LOG(WARNING) << layers_[i]->layer_param().name() << " ni=" << ni << " no=" << no;
      }

      //need to add it as cfg option :FIX_ME:SN
      const bool no_sparsity_for_small_kernel = true;
      bool need_sparsity_for_this_layer = true;
      if (no_sparsity_for_small_kernel) {
        need_sparsity_for_this_layer = (kernel_shape_data[0] > 2) && (kernel_shape_data[1] > 2) && (num_group != no);
      }

      //apply sparsity only to certain layers. exclude layers with small number of input and outputs
      //also exclude depth-wise separable layers.
      if((ni>=32 || no >= 32)  && (num_group<no) && need_sparsity_for_this_layer) {
        float threshold_fraction_selected = ((ni>=256 && no >= 512)? threshold_fraction_high :
            ((ni>=32 && no >= 32)? threshold_fraction_mid: threshold_fraction_low));

        for(int c=0; c<no; c++) {
          int weight_count_channel = ni * kernel_shape_data[0] * kernel_shape_data[1] / num_group;
          int start_index = weight_count_channel * c;

          float max_abs = std::abs(conv_weights.max(start_index, weight_count_channel));
          float min_abs = std::abs(conv_weights.min(start_index, weight_count_channel));
          float max_abs_value = std::max<float>(max_abs, min_abs);
          float step_size = max_abs_value * threshold_step_factor;
          float max_threshold_value = std::min<float>(std::min<float>(threshold_value_max, max_abs_value*threshold_value_maxratio), max_abs_value);
          bool verbose_th_val = false;
          if(verbose && verbose_th_val) {
            if ((max_abs_value*threshold_value_maxratio) > threshold_value_max) {
                LOG(INFO) << "threshold_value_max " << threshold_value_max;
                LOG(INFO) << "threshold_value_maxratio " << threshold_value_maxratio;
                LOG(INFO) << "max_abs_value*threshold_value_maxratio " << (max_abs_value*threshold_value_maxratio);
                LOG(INFO) << "final threshold_value used" << max_threshold_value; 
            }
          }

          float selected_threshold = 0;
          float granurality_start = 1000;
          for(float granurality = granurality_start, search_iter=0; granurality>=1; granurality=granurality/10, search_iter++) {
            float step_sizeX = step_size * granurality;
            float range_sizeX = step_sizeX*10*2;
            float start_valueX = selected_threshold;

            float min_step_val = search_iter>0? std::max((start_valueX-range_sizeX),0.0f) : 0;
            float max_step_val = search_iter>0? (start_valueX+range_sizeX) : max_threshold_value;
            for(float step= min_step_val; step<max_step_val && step<max_threshold_value; step+=step_sizeX) {
              float zcount = conv_weights.count_zero((float)step, start_index, weight_count_channel);
              float zratio = zcount / weight_count_channel;
              if(zratio <= threshold_fraction_selected) {
                selected_threshold = step;
              } else {
                break;
              }
            }
          }

          conv_weights.zerout(selected_threshold, start_index, weight_count_channel);
          //LOG(INFO) << "Layer:" << layer_name << " channel:" << c << " threshold:"
          //   << selected_threshold << " sparsity:"<< conv_weights.count_zero(0.0, start_index, weight_count_channel);
        }

        if(verbose) {
          float zcount = conv_weights.count_zero(0.0, 0, 0);
          LOG(WARNING) << layers_[i]->layer_param().name()
              //<< " MaxAbsWeight=" << max_abs_value
              //<< " MaxThreshold=" << max_threshold_value << " SelectedThreshold=" << selected_threshold
              << " ZeroWeightsFraction=" << (zcount/count);
        }
      }
    }
  }
}


/**
 * ApplySparseModeConnectivity
 * Yet another way to do this is to store the threshold for each layer in FindAndApplyThresholdNet
 * And just use it here. But the current implementation of this cuntion is more generic
 * since it can be used when thresholding is completely outside.
 */
void Net::ApplySparseModeConnectivity() {
  for (int i = 0; i < layers_.size(); i++) {
    if (layers_[i]->type() == std::string("Convolution")) {
      LayerBase& conv_layer = *layers_[i];
      Blob& conv_weights = *conv_layer.blobs()[0];

      //Use the connectivity information in the blob and zerout values accordingly.
      conv_weights.ComputeSparseData();

      //This is strictly not necessary
      //conv_weights.ComputeSparseDiff();
    }
  }
}

void Net::StoreSparseModeConnectivity(SparseMode mode) {
  LOG_IF(INFO, Caffe::root_solver()) << "All zero weights of convolution layers are frozen";
  if(mode != SPARSE_NONE) {
    for(int i=0; i<layers_.size(); i++) {
      if(layers_[i]->type() == std::string("Convolution")) {
        LayerBase& conv_layer = *layers_[i];
        Blob& conv_weights = *conv_layer.blobs()[0];

        //Store the non-zero weight information
        conv_weights.StoreSparseModeConnectivity(mode);
      }
    }
  }
}

float Net::DisplaySparsity(bool verbose) {
  float total_zero_count = 0, total_count = 0;
  {
    std::map<std::string, std::pair<int,int> > spasity_map;
    int blob_count = this->GetSparsity(spasity_map);
    if(verbose) {
      LOG(INFO) << "Num Params(" << blob_count << "), " << "Sparsity (zero_weights/count): ";
    }

    for(std::map<std::string, std::pair<int,int> >::iterator
        iter = spasity_map.begin(); iter != spasity_map.end(); iter++) {
      std::string param_name = iter->first;
      float zero_count = iter->second.first;
      float count = iter->second.second;
      total_zero_count += zero_count;
      total_count += count;
      if(verbose) {
        LOG(INFO) << param_name << "(" << std::setprecision(3) << (zero_count/count) << ") ";
      }
    }
    if(verbose) {
      LOG(INFO) << "Total Sparsity (zero_weights/count) = "
          << " (" << total_zero_count << "/" << total_count << ") "
          << std::setprecision(3) << (total_zero_count/total_count);
    }
  }

  return (total_zero_count/total_count);
}

float Net::DisplayConnectivitySparsity(bool verbose) {
  float total_zero_count = 0, total_count = 0;

  std::map<std::string, std::pair<int,int> > spasity_map;
  int blob_count = this->GetConnectivitySparsity(spasity_map);
  if(verbose) {
    LOG(INFO) << "Num Params(" << blob_count << "), " << "ConnectivitySparsity (zero_weights/count): ";
  }

  for(std::map<std::string, std::pair<int,int> >::iterator
      iter = spasity_map.begin(); iter != spasity_map.end(); iter++) {
    std::string param_name = iter->first;
    float zero_count = iter->second.first;
    float count = iter->second.second;
    total_zero_count += zero_count;
    total_count += count;
    if(verbose) {
      LOG(INFO) << param_name << "(" << std::setprecision(3) << (zero_count/count) << ") ";
    }
  }
  if(verbose) {
    LOG(INFO) << "Total ConnectivitySparsity (zero_weights/count) = "
        << " (" << total_zero_count << "/" << total_count << ") "
        << std::setprecision(3) << (total_zero_count/total_count);
  }

  return (total_zero_count/total_count);
}

int Net::GetSparsity(std::map<std::string, std::pair<int,int> >& sparsity_map){
  int blob_count = 0;
  float threshold = 0.0f;
  sparsity_map.clear();
  int max_params_to_check = 1;
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      const LayerParameter& layer_param = layers_[layer_id]->layer_param();

      bool next_layer_is_softmax = false;
      if((layer_id+1) < layers_.size() && (layers_[layer_id+1]->layer_param().type() == "Softmax" ||
          layers_[layer_id+1]->layer_param().type() == "SoftmaxWithLoss")) {
        next_layer_is_softmax = true;
      }
      bool next_layer_is_not_softmax = (!next_layer_is_softmax);
      bool is_candidate_layer = (layer_param.type() == "Convolution" /*|| layer_param.type() == "InnerProduct"*/);

      if(next_layer_is_not_softmax && is_candidate_layer)  {
          int num_params_to_check = std::min<int>(max_params_to_check, layers_[layer_id]->blobs().size());
          for (int param_id = 0; param_id < num_params_to_check;++param_id) {
            const Blob& blob = *layers_[layer_id]->blobs()[param_id];
            const int net_param_id = param_id_vecs_[layer_id][param_id];
            const string& blob_name = param_display_names_[net_param_id];
            //const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
            std::pair<int,int> sp_map = std::make_pair(blob.count_zero(threshold, 0, 0), blob.count());
            sparsity_map[layer_names_[layer_id] + "_param_" + blob_name] = sp_map;
            blob_count++;
          }
      }
  }
  return blob_count;
}

int Net::GetConnectivitySparsity(std::map<std::string, std::pair<int,int> >& sparsity_map){
  int blob_count = 0;
  float threshold = 0.0f;
  sparsity_map.clear();
  int max_params_to_check = 1;
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      const LayerParameter& layer_param = layers_[layer_id]->layer_param();

      bool next_layer_is_softmax = false;
      if((layer_id+1) < layers_.size() && (layers_[layer_id+1]->layer_param().type() == "Softmax" ||
          layers_[layer_id+1]->layer_param().type() == "SoftmaxWithLoss")) {
        next_layer_is_softmax = true;
      }
      bool next_layer_is_not_softmax = (!next_layer_is_softmax);
      bool is_candidate_layer = (layer_param.type() == "Convolution" /*|| layer_param.type() == "InnerProduct"*/);

      if(next_layer_is_not_softmax && is_candidate_layer) {
          int num_params_to_check = std::min<int>(max_params_to_check, layers_[layer_id]->blobs().size());
          for (int param_id = 0; param_id < num_params_to_check;++param_id) {
            const Blob& blob = *layers_[layer_id]->blobs()[param_id];
            const int net_param_id = param_id_vecs_[layer_id][param_id];
            const string& blob_name = param_display_names_[net_param_id];
            //const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
            std::pair<int,int> sp_map = std::make_pair(blob.count_zero_connectivity(threshold, 0, 0), blob.count());
            sparsity_map[layer_names_[layer_id] + "_param_" + blob_name] = sp_map;
            blob_count++;
          }
      }
  }
  return blob_count;
}

template void Net::Convert2FixedPoint_cpu(float* data, const int cnt, const int bw, int fl, bool unsigned_data, bool clip) const;

}  // namespace caffe
